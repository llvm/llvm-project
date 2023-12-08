OBJC_SOURCES := myclass.m repro.m
LD_EXTRAS := -lobjc

include Makefile.rules

# Force myclass.m to be compiled without debug info
myclass.o: CFLAGS = $(CFLAGS_NO_DEBUG)

