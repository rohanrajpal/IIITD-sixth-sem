import csv
import sys
with open('Kotak-NAV-History.csv') as csvfile:
    navreader = csv.reader(csvfile, delimiter=',')
    numstocks = 0
    sip_amount = 10000
    sell_date = sys.argv[1]
    invested_amount = 0
    next(navreader)
    for row in navreader:
        nav = float(row[2])
        if int(row[1][:2]) == 8:
            numstocks += sip_amount/nav
            invested_amount += sip_amount
        if row[1] == sell_date:
            sell_price = nav
            print(sell_price)
            break
    selling_amount = numstocks*sell_price
    total_returns = selling_amount - invested_amount
    total_returns_frac = total_returns/invested_amount
    print(f"Invested amount {invested_amount:,d}")
    print(f"Selling amount {selling_amount:,.2f}")
    print(f"Total returns {total_returns:,.2f}")
    print(f"Total % returns {100*total_returns_frac:,.2f}%")
    # print((1+total_returns_frac)**(1/3))
    print(f"Annualised returns {100*((1+total_returns_frac)**(1/3)-1)}%")
    
