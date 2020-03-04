LEVEL = ../../../make
EXE := main
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(shell pwd)
LD_EXTRAS = -lmod -L$(shell pwd)
include $(LEVEL)/Makefile.rules
