# We need to relocate source files, so we have to build from the generated build
# dir. We rely on the test script to copy the source files to the build dir.
CXX_SOURCES = $(BUILDDIR)/main.cpp
# So main.cpp can include "subdir/foo.h"
CXXFLAGS_EXTRAS = -I$(BUILDDIR)/subdir
LD_EXTRAS = $(BUILDDIR)/foo.o

include Makefile.rules

# Extra rules to make nested subdir sources compile.
$(EXE): foo.o

foo.o: $(BUILDDIR)/subdir/foo.cpp
	cd $(BUILDDIR) && $(CXX) $(CXXFLAGS) -MT $@ -MD -MP -MF foo.d -c -o $@ $<
