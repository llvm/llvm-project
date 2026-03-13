SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFT_ENABLE_COMPILE_CACHE := YES
SWIFT_BRIDGING_HEADER := bridging.h
HIDE_SWIFTMODULE := YES
SWIFTFLAGS_EXTRAS = -Xcc -I$(BUILDDIR)/secret -scanner-prefix-map-paths $(BUILDDIR) /^build

all: cas-config secret.h $(EXE)

include Makefile.rules

.PHONY: secret.h

secret.h:
	mkdir -p $(BUILDDIR)/secret
	echo "#ifndef SECRET_H" > $(BUILDDIR)/secret/secret.h
	echo "struct S { int i; };" >> $(BUILDDIR)/secret/secret.h
	echo "#endif" >> $(BUILDDIR)/secret/secret.h

cas-config:
	echo "{\"CASPath\": \"$(BUILDDIR)/cas\"}" > .cas-config
