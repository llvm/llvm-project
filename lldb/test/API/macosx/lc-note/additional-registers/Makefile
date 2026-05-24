MAKE_DSYM := NO
C_SOURCES := main.c
CXXFLAGS_EXTRAS := -std=c++17

all: a.out add-lcnote

add-lcnote:
	"$(MAKE)" -f "$(MAKEFILE_RULES)" EXE=add-lcnote \
	    CXX=$(CC) CXXFLAGS_EXTRAS="$(CXXFLAGS_EXTRAS)" CXX_SOURCES=add-lcnote.cpp

include Makefile.rules
