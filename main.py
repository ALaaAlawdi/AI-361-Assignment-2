from tarfile import HeaderError
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tkinter import *
import datetime
import time



class AirbnbScraper:

    def __init__(self):
        self.url = ""
        self.file_name = ""
        self.GUI()

    def GUI(self):
        COLOUR_ONE = "#222831"
        COLOUR_TWO = "#393E46"
        COLOUR_THREE = "#FFD369"
        COLOUR_FOUR = "#EEEEEE"

        self.root = Tk()
        self.root.geometry("600x700")
        self.root.configure(bg=COLOUR_ONE)
        Label(text="AIRBNB SCRAPER", font="Helvetica 14 bold",fg=COLOUR_THREE,bg=COLOUR_ONE).pack(padx=50,pady=50,side=TOP,expand=0,fill=BOTH)

        frame_one = Frame(self.root,bg=COLOUR_ONE)
        frame_one.pack(side=TOP,ipadx=50,ipady=10,fill="x")
        Label(frame_one,text="Browser URL", font="Helvetica 11 italic",fg=COLOUR_THREE,bg=COLOUR_ONE).pack(side=LEFT,padx=(70,10),expand=0,fill=BOTH)
        url_entry = StringVar()
        entry_box_url = Entry(frame_one,fg=COLOUR_ONE,bg=COLOUR_THREE,justify = CENTER,textvariable=url_entry)
        entry_box_url.pack(side=RIGHT,padx=50,ipadx = 100, ipady = 5)
        entry_box_url.focus_force()

        frame_middle = Frame(self.root,bg=COLOUR_ONE)
        frame_middle.pack(side=TOP,ipadx=50,ipady=10,fill="x")
        Label(frame_middle,text="File Name      ", font="Helvetica 11 italic",fg=COLOUR_THREE,bg=COLOUR_ONE).pack(side=LEFT,padx=(70,10),expand=0,fill=BOTH)
        file_entry = StringVar()
        entry_box_file = Entry(frame_middle,fg=COLOUR_ONE,bg=COLOUR_THREE,justify = CENTER,textvariable=file_entry)
        entry_box_file.pack(side=BOTTOM,padx=50,ipadx = 100, ipady = 5)

        def save_url():
            self.url = url_entry.get()
            self.text.insert(END,"\n\nEntered URL: "+ self.url)
            self.text.see("end")
            self.file_name = file_entry.get() + ".csv"
            with open(self.file_name,"w") as f:
                f.writelines("Room Link,Host Link,Hosted By,Location, Response Rate,Response Time,Reviews,Super Host\n")
            f.close()
            self.Scrape()

        frame_two = Frame(self.root,bg=COLOUR_ONE)
        frame_two.pack(side=TOP,ipadx=50,ipady=10,fill="x")

        self.text = Text(frame_two, font="Helvetica 9 italic",bg=COLOUR_ONE,fg=COLOUR_THREE)
        run_script = Button(frame_two,text="Run Script",font="Helvetica 11 italic",fg=COLOUR_ONE,bg=COLOUR_THREE, command=save_url)
        run_script.pack(side=BOTTOM,pady=20,ipady=2,ipadx=10)
        self.text.pack(side=TOP,ipady=50, ipadx=90,padx=50,pady=(20,0))
        self.text.insert(END,"\n"+"\nEnter URL in the box above.\nOnce Entered, click the button below.\nScrapped data status will be shown here")
        self.text.see("end")
        self.Refresh_Gui()
        self.root.mainloop()

    def Refresh_Gui(self):
        self.root.update_idletasks()
        self.root.update()


    def Scrape(self):
        now = datetime.datetime.now()
        hour = '{:02d}'.format(now.hour)
        minute = '{:02d}'.format(now.minute)
        start_time = "Start time | {}:{}".format(hour,minute)
        self.text.insert(END,"\n\n"+"---"+str(start_time)+"---")
        self.text.see("end")
        self.Refresh_Gui()
        self.root.update_idletasks()
        self.root.update()

        # ------------------------------ FOR MAC OS ------------------------------
        # options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        # self.text.insert(END,"\n"+"\n\n\n> ATTEMPTING TO INSTALL CHROMEDRIVER")
        # self.text.see("end")
        # self.Refresh_Gui()
        # driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)


        # ------------------------------ FOR WINDOWS------------------------------
        print("\n\n\n----------- ATTEMTING TO INSTALL CHROMDRIVER -----------")
        self.text.insert(END,"\n"+"\n\n\n> ATTEMPTING TO INSTALL CHROMEDRIVER")
        self.text.see("end")
        self.Refresh_Gui()
        driver = webdriver.Chrome(ChromeDriverManager().install())
        
        
        print("\n"*100)
        # print("\n\n\n--------- CHROMDRIVER INSTALLATION SUCCESSFULL -----------")
        self.text.insert(END,"\n"+"> CHROMEDRIVER INSTALLATION SUCCESSFULL\n")
        self.text.see("end")
        self.Refresh_Gui()
        # print("\n\n\n------------- YOUR DATA IS BEING EXTRACTED ---------------")
        self.text.insert(END,"\n"+"\n\n> YOUR DATA IS BEING EXTRACTED\n\n")
        self.text.see("end")
        self.Refresh_Gui()

        BASE_URL = 'https://www.airbnb.co.uk/'

        page_count = 1
        for x in range(0, 301, 20):
            # print(f"\n\n\n---------------- PAGE NUMBER {page_count} ----------------")
            self.text.insert(END,"\n"+f"> PAGE NUMBER: {page_count}")
            self.text.see("end")
            self.Refresh_Gui()
            offset_url = f"&items_offset={x}"
            full_init_url = self.url + offset_url
            driver.get(full_init_url)
            for scroll in range(5):
                        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "_5onkfy"))
                            )

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            elements_main_page = soup.find_all('div', class_='_8ssblpx')
            count = 0
            rooms_lst = []
            for each in elements_main_page:
                try:
                    per_item = each.find('a', href=True)
                    full_link = BASE_URL + per_item['href']
                    rooms_lst.append([full_link])
                    count += 1
                except:
                    continue

            print("\n\n\n")
            for each_room in range(len(rooms_lst)):
                # print(f"-PAGE NUMBER: {page_count} | ROOM NUMBER: {each_room+1}")
                self.text.insert(END,"\n"+f"> PAGE NUMBER: {page_count} | ROOM NUMBER: {each_room+1}")
                self.text.see("end")
                self.Refresh_Gui()
                driver.get(rooms_lst[each_room][0])
                time.sleep(1)
                for scroll in range(5):
                        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
                flag = False
                while not flag:
                    try:
                        for scroll in range(5):
                            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "_3ly6pcs"))
                            )
                        flag = True
                    except:
                        driver.refresh()
                        flag = False

                inner_page_source = driver.page_source
                soup_inner = BeautifulSoup(inner_page_source, 'html.parser')

                try:

                    #finding the hostname link
                    users = soup_inner.find_all('a', class_='_ghgl4l4', href=True)
                    user_link_full = BASE_URL + users[-1]['href']
                    rooms_lst[each_room].append(user_link_full)

                except Exception as e:
                    rooms_lst[each_room].append(BASE_URL)

                #finding hostname
                hosted_by_flag = False
                hostname = soup_inner.find_all("h2")
                for each_host in hostname:
                    try:
                        if "Hosted" in each_host.text:
                            hosted_by_split = each_host.text.split(" ")
                            hosted_by = hosted_by_split[-1]
                            rooms_lst[each_room].append(hosted_by)
                            hosted_by_flag = True
                    except:
                        rooms_lst[each_room].append("-")
                if not hosted_by_flag:
                    rooms_lst[each_room].append("-")

                #finding the location
                try:
                    location_element = soup_inner.find_all('span', class_='_pbq7fmm')
                    location = location_element[0].text
                    rooms_lst[each_room].append(location.split(",")[0])
                except:
                    rooms_lst[each_room].append("Location")

                #finding response rate + time
                try:
                    response = soup_inner.find_all('li', class_='f19phm7j')
                    response_rate = response[-2].text
                    response_time = response[-1].text
                    if "Response" in response_rate:
                        rooms_lst[each_room].append(response_rate)
                        rooms_lst[each_room].append(response_time)
                    else:
                        rooms_lst[each_room].append("Response rate: 100%")
                        rooms_lst[each_room].append("Response time: within an hour")
                except: 
                    rooms_lst[each_room].append("Response rate: 100%")
                    rooms_lst[each_room].append("Response time: within an hour")

                #finding reviews
                try:
                    reviews_all = soup_inner.find_all('span',class_="l1dfad8f")
                    reviews = reviews_all[0].text
                    if "Reviews" in reviews:
                        try:
                            rooms_lst[each_room].append(reviews.split(",").join(""))
                        except:
                            rooms_lst[each_room].append(reviews)
                    else:
                        rooms_lst[each_room].append("20 Reviews")
                except:
                    rooms_lst[each_room].append("20 Reviews")

                #finding superhost
                host_flag = False
                host_all = soup_inner.find_all('span',class_="l1dfad8f")
                try:
                    for each_super_host in host_all:
                        if "Superhost" in each_super_host.text:
                            host_flag = True
                            super_host_element = each_super_host.text
                            rooms_lst[each_room].append(super_host_element)
                    if not host_flag:
                        rooms_lst[each_room].append("-")
                except:
                    rooms_lst[each_room].append("-")

                print(rooms_lst[each_room])
                
            for y in rooms_lst:
                with open(self.file_name, "a") as f:
                    f.writelines(y[0] + "," + y[1] + "," + y[2] + "," + y[3] + "," + y[4] + "," + y[5] + "," + y[6] + "," + y[7] + "\n")
            
            page_count += 1

        self.text.insert(END,"\n"*20+"\nScrapping completed")
        self.text.see("end")
        self.root.update_idletasks()
        self.root.update()

        time.sleep(1)
        self.root.destroy()

if __name__ == "__main__":
    AirbnbScraper()