CXX_SOURCES := main.cpp
CXXFLAGS_EXTRAS := -std=c++14

ifneq (,$(findstring icc,$(CC)))
    CXXFLAGS_EXTRAS := -debug inline-debug-info
endif


include Makefile.rules
