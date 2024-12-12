CXX_SOURCES := main.cpp
LD_EXTRAS := ns1.o ns2.o ns3.o ns4.o

a.out: main.o ns1.o ns2.o ns3.o ns4.o

ns1.o: common.cpp
	$(CC) -gdwarf -c -DNAMESPACE=ns1 -o $@ $<

ns2.o: common.cpp
	$(CC) -gdwarf -c -DNAMESPACE=ns2 -o $@ $<

ns3.o: common.cpp
	$(CC) -gdwarf -c -DNAMESPACE=ns3 -o $@ $<

ns4.o: common.cpp
	$(CC) -gdwarf -c -DNAMESPACE=ns4 -o $@ $<


include Makefile.rules
