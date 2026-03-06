SWIFT_SOURCES := main.swift

all: fakesdk a.out

include Makefile.rules

fakesdk:
	$(call LN_SF,$(SDKROOT),$@)

SWIFTFLAGS := $(subst $(SDKROOT),$(shell pwd)/fakesdk,$(SWIFTFLAGS))

clean::
	$(call RM_F,fakesdk a.out *.dSYM)
