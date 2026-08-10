C_SOURCES := main.c
CFLAGS_EXTRAS = -g -I$(SRCDIR)
LD_EXTRAS = $(BUILDDIR)/cached/libcached.dylib $(BUILDDIR)/uncached/libuncached.dylib

all: a.out

include Makefile.rules

# Two dylibs built from the same headers. The cached one is compiled through
# the explicit-module driver with a CAS, so its DWARF refers to the .pcm by
# CASID; the uncached one is built the ordinary way, so its DWARF refers to the
# .pcm by path. Both get a .cas-config next to them pointing at a CAS of their
# own, so both resolve to a CAS -- but only the cached one loads a module out of
# one.
#
# Built with explicit rules rather than DYLIB_C_SOURCES because the cached
# dylib needs a bespoke compile step and neither may be run through dsymutil:
# that would resolve the -gmodules references and there would be no module
# left to load out of the CAS.
#
# The build directory is not erased between test runs, so each rule spells out
# everything it is built from, the tools included: a rebuilt clang has to
# rebuild the dylibs, or the module lldb reads out of the CAS would be one the
# previous clang wrote.

CAS_DIR = $(BUILDDIR)/cached
NOCAS_DIR = $(BUILDDIR)/uncached
MODULE_FLAGS = -fmodules -gmodules -I$(SRCDIR)
COMPILE_FLAGS = -g $(SYSROOT_FLAGS) $(ARCH_CFLAGS)
EXPLICIT_MODULE_BUILD = $(LLDB_BASE_DIR)/scripts/clang-explicit-module-build.py
CLANG_SCAN_DEPS = $(dir $(firstword $(CC)))clang-scan-deps

a.out: $(CAS_DIR)/libcached.dylib $(NOCAS_DIR)/libuncached.dylib

# The explicit-module build shells out to clang-scan-deps next to $(CC), so that
# is a tool this rule uses too.
$(CAS_DIR)/libcached.dylib: $(SRCDIR)/cached.c $(SRCDIR)/cached.h \
                            $(SRCDIR)/module.modulemap \
                            $(EXPLICIT_MODULE_BUILD) \
                            $(call tool_prereq,$(CC)) \
                            $(call tool_prereq,$(CLANG_SCAN_DEPS)) \
                            | $(CAS_DIR)/.cas-config
	python3 $(EXPLICIT_MODULE_BUILD) -cas-path $(CAS_DIR)/cas -- \
	  $(CC) -c $(COMPILE_FLAGS) $(MODULE_FLAGS) \
	  -fmodules-cache-path=$(CAS_DIR)/module-cache \
	  -o $(CAS_DIR)/cached.o $(SRCDIR)/cached.c
	$(CC) -dynamiclib $(COMPILE_FLAGS) $(CAS_DIR)/cached.o -o $@

$(NOCAS_DIR)/libuncached.dylib: $(SRCDIR)/uncached.c $(SRCDIR)/uncached.h \
                                $(SRCDIR)/module.modulemap \
                                $(call tool_prereq,$(CC)) \
                                | $(NOCAS_DIR)/.cas-config
	$(CC) -c $(COMPILE_FLAGS) $(MODULE_FLAGS) \
	  -fmodules-cache-path=$(NOCAS_DIR)/module-cache \
	  -o $(NOCAS_DIR)/uncached.o $(SRCDIR)/uncached.c
	$(CC) -dynamiclib $(COMPILE_FLAGS) $(NOCAS_DIR)/uncached.o -o $@

# One rule per config file rather than an echo in the recipes above, so that a
# config that is missing from an existing build directory comes back without
# the dylib next to it having to be rebuilt. An order-only prerequisite above:
# what a dylib is built from does not depend on this file's contents, only on
# the directory existing.
$(CAS_DIR)/.cas-config $(NOCAS_DIR)/.cas-config:
	$(call MKDIR_P,$(@D))
	echo '{"CASPath": "$(@D)/cas"}' > $@
