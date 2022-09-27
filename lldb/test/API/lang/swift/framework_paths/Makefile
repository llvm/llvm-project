SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS = -F $(BUILDDIR) -framework Direct -Xlinker -rpath -Xlinker $(BUILDDIR)

all: Direct.framework $(EXE)

include Makefile.rules

Discovered.framework: Discovered.h
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Discovered \
		DYLIB_OBJC_SOURCES=Discovered.m \
		FRAMEWORK_HEADERS=$(SRCDIR)/Discovered.h \
		FRAMEWORK_MODULES=$(SRCDIR)/module.modulemap \
		FRAMEWORK=Discovered

Direct.framework: $(SRCDIR)/Direct.swift.in Discovered.framework
	mkdir -p $(BUILDDIR)/secret_path
	cp $< $(BUILDDIR)/Direct.swift
	mv Discovered.framework $(BUILDDIR)/secret_path
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Direct \
		DYLIB_SWIFT_SOURCES=Direct.swift \
		DYLIB_MODULENAME=Direct \
		FRAMEWORK=Direct \
		SWIFTFLAGS_EXTRAS=-F$(BUILDDIR)/secret_path
	rm -f $(BUILDDIR)/Direct.swiftmodule $(BUILDDIR)/Direct.swiftinterface $(BUILDDIR)/Direct.swift
