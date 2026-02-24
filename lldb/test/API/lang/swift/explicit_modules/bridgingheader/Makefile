SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFT_BRIDGING_HEADER := bridging.h
SWIFTFLAGS_EXTRAS = -Xcc -I$(BUILDDIR)/secret

all: secret.h $(EXE)

include Makefile.rules

.PHONY: secret.h

secret.h:
	mkdir -p $(BUILDDIR)/secret
	echo "#ifndef SECRET_H" > $(BUILDDIR)/secret/secret.h
	echo "struct S { int i; };" >> $(BUILDDIR)/secret/secret.h
	echo "#endif" >> $(BUILDDIR)/secret/secret.h
