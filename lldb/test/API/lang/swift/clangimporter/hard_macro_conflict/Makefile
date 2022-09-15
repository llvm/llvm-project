SWIFT_SOURCES = main.swift
SWIFT_OBJC_INTEROP := 1

all: Framework.framework a.out

include Makefile.rules
BAR_FLAGS= -Xcc -I$(SRCDIR)/Bar -Xcc -DMACRO=struct
FOO_FLAGS= -Xcc -I$(SRCDIR)/Foo -Xcc -DMACRO=int
SWIFTFLAGS_EXTRAS=$(BAR_FLAGS) \
  -F$(BUILDDIR) -framework Framework -L$(BUILDDIR) \
  -Xlinker -rpath -Xlinker $(BUILDDIR)

Framework.framework: Framework.swift
	$(MAKE) -f $(MAKEFILE_RULES) -C $(BUILDDIR) VPATH=$(VPATH) \
                MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=$(shell basename $< .swift) \
		DYLIB_ONLY=YES DYLIB_SWIFT_SOURCES=Framework.swift \
		SWIFT_OBJC_INTEROP=1 \
		SWIFTFLAGS_EXTRAS="$(FOO_FLAGS)" \
		FRAMEWORK=Framework
	rm -f $(BUILDDIR)/Framework.swiftmodule
	ln -s $(BUILDDIR)/Framework.framework/Framework $(BUILDDIR)/Framework # FIXME
