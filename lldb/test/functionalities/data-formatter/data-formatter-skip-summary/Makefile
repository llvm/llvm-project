LEVEL = ../../../make

CXX_SOURCES := main.cpp

include $(LEVEL)/Makefile.rules

CXXFLAGS += -O0

ifeq (,$(findstring gcc,$(CC)))
CXXFLAGS += -stdlib=libstdc++
LDFLAGS += -stdlib=libstdc++
endif
