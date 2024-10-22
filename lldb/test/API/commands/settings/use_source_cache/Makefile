CXX_SOURCES := main-copy.cpp

include Makefile.rules


# Copy file into the build folder to enable the test to modify it.
main-copy.cpp: main.cpp
	cp -f $< $@
ifneq "$(OS)" "Windows_NT"  # chmod is not available on Windows
	chmod u+w $@
endif
