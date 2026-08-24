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

# Prerequisite lists are expanded as the rule is read, so any rule using
# tool_prereq has to come after the include that defines it.
include Makefile.rules

# Artificially inject a wrong SDK into a C file to test that it is being ignored.
ignored.o: ignored.c \
           $(call tool_prereq,$(CC))
	$(CC) -target $(ARCH)-apple-macos  -c $< -o $@ \
	  -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path)
