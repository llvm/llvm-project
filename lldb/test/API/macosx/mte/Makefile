C_SOURCES := main.c

EXE := uaf_mte

all: uaf_mte sign

include Makefile.rules

sign: mte-entitlements.plist uaf_mte
ifeq ($(OS),Darwin)
	codesign -s - -f --entitlements $^
endif
