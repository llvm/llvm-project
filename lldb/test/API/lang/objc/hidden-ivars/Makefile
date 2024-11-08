DYLIB_NAME := InternalDefiner
DYLIB_OBJC_SOURCES := InternalDefiner.m
OBJC_SOURCES := main.m

LD_EXTRAS = -framework Foundation

all: a.out libInternalDefiner.dylib stripped

include Makefile.rules

ifeq "$(MAKE_DSYM)" "YES"
stripped: a.out.dSYM
endif

stripped: a.out libInternalDefiner.dylib
	mkdir stripped
	$(STRIP) -Sx a.out -o stripped/a.out
	$(STRIP) -Sx libInternalDefiner.dylib -o stripped/libInternalDefiner.dylib
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - stripped/a.out
endif
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - stripped/libInternalDefiner.dylib
endif
ifeq "$(MAKE_DSYM)" "YES"
	cp -r a.out.dSYM stripped/a.out.dSYM
endif
