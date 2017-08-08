for i in {1..100};do ./simple >> simple.txt;./spawn >> spawn.txt;done;python average.py;rm *.txt
