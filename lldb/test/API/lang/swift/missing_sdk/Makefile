SWIFT_SOURCES := main.swift

all: fakesdk a.out

include Makefile.rules

fakesdk:
	ln -s $(SDKROOT) $@

SWIFTFLAGS := $(subst $(SDKROOT),$(shell pwd)/fakesdk,$(SWIFTFLAGS))

clean::
	rm -f fakesdk a.out *.dSYM
