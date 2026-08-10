CXX_SOURCES := main.cpp
CXXFLAGS_EXTRAS := -std=c++14

ifeq ($(CC_TYPE), icc)
    CXXFLAGS_EXTRAS := -debug inline-debug-info
endif


include Makefile.rules
