CXX_SOURCES := relative.cpp

EXE := CompDirSymLink

include Makefile.rules

# Force relative filenames by copying it into the build directory.
relative.cpp: main.cpp
	cp -f $< $@
