SWIFT_SOURCES := main.swift
ifeq "$(TRIPLE)" ""
SWIFTFLAGS_EXTRAS := -Xfrontend -no-serialize-debugging-options
else
# Target override for the simulator testcase.
SWIFTFLAGS_EXTRAS := -Xfrontend -no-serialize-debugging-options -target $(TRIPLE)
endif
SWIFT_BRIDGING_HEADER := bridging-header.h
LD_EXTRAS := ignored.o

all: ignored.o $(EXE)

# Artificially inject a wrong SDK into a C file to test that it is being ignored.
ignored.o: ignored.c
	$(CC) -target $(ARCH)-apple-macos  -c $< -o $@ \
	  -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path)

include Makefile.rules
