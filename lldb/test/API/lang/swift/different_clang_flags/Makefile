EXE := main
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR) -I$(SRCDIR)/moda_dir/cdeps -I$(SRCDIR)/modb_dir/cdeps
LD_EXTRAS = -L$(BUILDDIR) -lmoda -lmodb

all: libmoda.dylib libmodb.dylib main

include Makefile.rules

lib%.dylib:
	$(MAKE) -f $(MAKEFILE_RULES) VPATH=$(VPATH)/$*_dir \
		SWIFTFLAGS_EXTRAS=-I$(dir $<)/cdeps \
                MAKE_DSYM=YES DYLIB_ONLY=YES \
		DYLIB_NAME=$* DYLIB_SWIFT_SOURCES=$*.swift \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -I$(SRCDIR)/moda_dir/cdeps -I$(SRCDIR)/modb_dir/cdeps"
