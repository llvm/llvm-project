SWIFT_SOURCES := main.swift

all: fakesdk a.out

include Makefile.rules

fakesdk:
	$(call LN_SF,$(SDKROOT),$@)

# On platforms with no SDK (e.g. Linux) SDKROOT is empty, and $(subst) with an
# empty pattern appends the replacement to SWIFTFLAGS rather than substituting
# anything, corrupting the last flag. There is no SDK path to redirect there
# anyway, so only substitute when SDKROOT is set.
ifneq "$(SDKROOT)" ""
SWIFTFLAGS := $(subst $(SDKROOT),$(shell pwd)/fakesdk,$(SWIFTFLAGS))
endif

clean::
	$(call RM_F,fakesdk a.out *.dSYM)
