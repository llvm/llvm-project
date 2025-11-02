C_SOURCES := main.c

EXE := uaf

binary-plain:    uaf
binary-entitled: uaf sign

all: binary-entitled

include Makefile.rules

sign: mte-entitlements.plist uaf
ifeq ($(OS),Darwin)
	codesign -s - -f --entitlements $^
endif
