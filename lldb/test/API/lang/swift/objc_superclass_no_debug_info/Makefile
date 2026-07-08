SWIFT_SOURCES := main.swift

all: libMyObjCBase.dylib libMyLib.dylib a.out

include Makefile.rules

LD_EXTRAS = -lMyLib -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR) -I$(SRCDIR) -Xcc -fmodule-map-file=$(SRCDIR)/module.modulemap

libMyObjCBase.dylib: $(SRCDIR)/MyObjCBase.m $(SRCDIR)/MyObjCBase.h
	$(CC) -isysroot "$(SDKROOT)" -g0 -c $(SRCDIR)/MyObjCBase.m \
		-o $(BUILDDIR)/MyObjCBase.o
	$(CC) -isysroot "$(SDKROOT)" -dynamiclib -lobjc \
		-install_name @executable_path/libMyObjCBase.dylib \
		$(BUILDDIR)/MyObjCBase.o \
		-o $(BUILDDIR)/libMyObjCBase.dylib
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$(BUILDDIR)/libMyObjCBase.dylib"
endif

libMyLib.dylib: $(SRCDIR)/MyLib.swift libMyObjCBase.dylib
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		VPATH=$(SRCDIR) \
		MAKE_DSYM=NO \
		DEBUG_INFO_FLAG= SWIFT_DEBUG_INFO_FLAG= \
		DYLIB_ONLY=YES DYLIB_NAME=MyLib \
		DYLIB_SWIFT_SOURCES=MyLib.swift \
		SWIFTFLAGS_EXTRAS="-I$(SRCDIR) -Xcc -fmodule-map-file=$(SRCDIR)/module.modulemap" \
		LD_EXTRAS="-lMyObjCBase -L$(BUILDDIR)"
