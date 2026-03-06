SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS = -I$(BUILDDIR) -F$(BUILDDIR)

FRAMEWORKS=$(patsubst %,a%.framework,$(shell seq -s " " 0 $(N)))

all: $(FRAMEWORKS) dylib $(EXE)

include Makefile.rules

lib%.a: %.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=static \
		SWIFT_SOURCES=$(patsubst lib%.a,%.swift,$@) \
		MODULENAME=$(patsubst lib%.a,%,$@) \
		SWIFTFLAGS_EXTRAS="$(patsubst %,-F%/DOES_NOT_EXIST,$(FRAMEWORKS))" \
		$(patsubst lib%.a,%.swift.o,$@) \
		$(patsubst lib%.a,%.swiftmodule,$@)
	ar -r $@ $(BUILDDIR)/$(patsubst lib%.a,%.swift.o,$@)

%.framework: lib%.a
	$(call MKDIR_P,$(BUILDDIR)/$@/Headers)
	$(call MKDIR_P,$(BUILDDIR)/$@/Modules)
	$(call MKDIR_P,$(BUILDDIR)/$@/Resources)
	mv $< $(BUILDDIR)/$@/$(patsubst lib%.a,%,$<)
	$(call MKDIR_P,$(BUILDDIR)/$@/Modules/$(patsubst lib%.a,%.swiftmodule,$<))
	mv $(BUILDDIR)/$(patsubst lib%.a,%.swiftmodule,$<) $(BUILDDIR)/$@/Modules/$(patsubst lib%.a,%.swiftmodule,$<)/$(ARCH)-apple-macos.swiftmodule
	mv $(BUILDDIR)/$(patsubst lib%.a,%.swiftinterface,$<) $(BUILDDIR)/$@/Modules/

dylib: Dylib.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Dylib \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_MODULENAME=Dylib \
		FRAMEWORK=Dylib
	$(call RM_F,$(BUILDDIR)/Dylib.swiftmodule $(BUILDDIR)/Dylib.swiftinterface)
