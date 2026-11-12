OBJCXX_SOURCES := main.mm myobject.mm
include Makefile.rules

# myobject.o needs to be built without debug info
myobject.o: myobject.mm
	$(CXX) $(CFLAGS_NO_DEBUG) -c -o $@ $<
