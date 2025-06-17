OBJC_SOURCES := main.m
CFLAGS_EXTRAS = $(MANDATORY_MODULE_BUILD_CFLAGS) -I. -I$(BUILDDIR)

SWIFT_SOURCES := Foo.swift
SWIFTFLAGS_EXTRAS = -Xcc -I$(SRCDIR) -emit-objc-header-path Foo.h -parse-as-library

all: Foo.swift.o $(EXE)

include Makefile.rules
