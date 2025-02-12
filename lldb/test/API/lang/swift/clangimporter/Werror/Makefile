SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -Xcc -DCONFLICT=1 -Xcc -Werror -I. -L. -lDylib

all: libDylib.dylib a.out

include Makefile.rules

libDylib.dylib: dylib.swift
	"$(MAKE)" BASENAME=$(shell basename $< .swift) \
		-f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o

