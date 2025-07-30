SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := Yes
SWIFT_BRIDGING_HEADER := bridging.h
SWIFTFLAGS_EXTRAS = -Xcc -I$(BUILDDIR)/secret

all: secret.h $(EXE)

include Makefile.rules

.PHONY: secret.h

secret.h:
	mkdir -p $(BUILDDIR)/secret
	echo "struct S { int i; };" > $(BUILDDIR)/secret/secret.h
