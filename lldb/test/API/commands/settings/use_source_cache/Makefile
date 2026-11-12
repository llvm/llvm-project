CXX_SOURCES := main-copy.cpp

include Makefile.rules


# Copy file into the build folder to enable the test to modify it.
main-copy.cpp: main.cpp
	cp -f $< $@
	chmod u+w $@
