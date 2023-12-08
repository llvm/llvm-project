CXX_SOURCES := main.cpp

ifneq (,$(findstring icc,$(CC)))
    CXXFLAGS_EXTRAS := -debug inline-debug-info
endif


include Makefile.rules
