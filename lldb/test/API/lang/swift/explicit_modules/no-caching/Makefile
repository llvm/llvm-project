SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES

all: a.out cas-config

include Makefile.rules

cas-config:
	echo "{\"CASPath\": \"$(BUILDDIR)/cas\"}" > .cas-config
