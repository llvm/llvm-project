LEVEL = ../../../make

OBJC_SOURCES := main.m

CFLAGS ?= -arch $(ARCH) -gdwarf-2 -O2
LDFLAGS = $(CFLAGS) -lobjc -framework Foundation

include $(LEVEL)/Makefile.rules
