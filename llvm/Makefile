#===- ./Makefile -------------------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

LEVEL := .

# Top-Level LLVM Build Stages:
#   1. Build lib/System and lib/Support, which are used by utils (tblgen).
#   2. Build utils, which is used by VMCore.
#   3. Build VMCore, which builds the Intrinsics.inc file used by libs.
#   4. Build libs, which are needed by llvm-config.
#   5. Build llvm-config, which determines inter-lib dependencies for tools.
#   6. Build tools, runtime, docs.
#
# When cross-compiling, there are some things (tablegen) that need to
# be build for the build system first.
ifeq ($(BUILD_DIRS_ONLY),1)
  DIRS := lib/System lib/Support utils
  OPTIONAL_DIRS :=
else
  DIRS := lib/System lib/Support utils lib/VMCore lib tools/llvm-config \
          tools runtime docs
  OPTIONAL_DIRS := examples projects bindings
endif

EXTRA_DIST := test unittests llvm.spec include win32 Xcode

include $(LEVEL)/Makefile.config

# llvm-gcc4 doesn't need runtime libs.  llvm-gcc4 is the only supported one.
# FIXME: Remove runtime entirely once we have an understanding of where
# libprofile etc should go.
#ifeq ($(LLVMGCC_MAJVERS),4)
  DIRS := $(filter-out runtime, $(DIRS))
#endif

ifeq ($(MAKECMDGOALS),libs-only)
  DIRS := $(filter-out tools runtime docs, $(DIRS))
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),install-libs)
  DIRS := $(filter-out tools runtime docs, $(DIRS))
  OPTIONAL_DIRS := $(filter bindings, $(OPTIONAL_DIRS))
endif

ifeq ($(MAKECMDGOALS),tools-only)
  DIRS := $(filter-out runtime docs, $(DIRS))
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),install-clang)
  DIRS := tools/clang/Driver tools/clang/lib/Headers tools/clang/tools/ccc
  OPTIONAL_DIRS :=
  NO_INSTALL = 1
endif

ifeq ($(MAKECMDGOALS),clang-only)
  DIRS := $(filter-out tools runtime docs, $(DIRS)) tools/clang
  OPTIONAL_DIRS :=
endif

ifeq ($(MAKECMDGOALS),unittests)
  DIRS := $(filter-out tools runtime docs, $(DIRS)) utils unittests
  OPTIONAL_DIRS :=
endif

# Don't install utils, examples, or projects they are only used to
# build LLVM.
ifeq ($(MAKECMDGOALS),install)
  DIRS := $(filter-out utils, $(DIRS))
  OPTIONAL_DIRS := $(filter bindings, $(OPTIONAL_DIRS))
endif

# If we're cross-compiling, build the build-hosted tools first
ifeq ($(LLVM_CROSS_COMPILING),1)
all:: cross-compile-build-tools

clean::
	$(Verb) rm -rf BuildTools

cross-compile-build-tools:
	$(Verb) if [ ! -f BuildTools/Makefile ]; then \
          $(MKDIR) BuildTools; \
	  cd BuildTools ; \
	  $(PROJ_SRC_DIR)/configure ; \
	  cd .. ; \
	fi; \
        ($(MAKE) -C BuildTools BUILD_DIRS_ONLY=1 ) || exit 1;
endif

# Include the main makefile machinery.
include $(LLVM_SRC_ROOT)/Makefile.rules

# Specify options to pass to configure script when we're
# running the dist-check target
DIST_CHECK_CONFIG_OPTIONS = --with-llvmgccdir=$(LLVMGCCDIR)

.PHONY: debug-opt-prof
debug-opt-prof:
	$(Echo) Building Debug Version
	$(Verb) $(MAKE)
	$(Echo)
	$(Echo) Building Optimized Version
	$(Echo)
	$(Verb) $(MAKE) ENABLE_OPTIMIZED=1
	$(Echo)
	$(Echo) Building Profiling Version
	$(Echo)
	$(Verb) $(MAKE) ENABLE_PROFILING=1

dist-hook::
	$(Echo) Eliminating files constructed by configure
	$(Verb) $(RM) -f \
	  $(TopDistDir)/include/llvm/ADT/hash_map.h  \
	  $(TopDistDir)/include/llvm/ADT/hash_set.h  \
	  $(TopDistDir)/include/llvm/ADT/iterator.h  \
	  $(TopDistDir)/include/llvm/Config/config.h  \
	  $(TopDistDir)/include/llvm/Support/DataTypes.h  \
	  $(TopDistDir)/include/llvm/Support/ThreadSupport.h

clang-only: all
tools-only: all
libs-only: all
install-clang: install
install-libs: install

#------------------------------------------------------------------------
# Make sure the generated headers are up-to-date. This must be kept in
# sync with the AC_CONFIG_HEADER invocations in autoconf/configure.ac
#------------------------------------------------------------------------
FilesToConfig := \
  include/llvm/Config/config.h \
  include/llvm/Support/DataTypes.h \
  include/llvm/ADT/hash_map.h \
  include/llvm/ADT/hash_set.h \
  include/llvm/ADT/iterator.h
FilesToConfigPATH  := $(addprefix $(LLVM_OBJ_ROOT)/,$(FilesToConfig))

all-local:: $(FilesToConfigPATH)
$(FilesToConfigPATH) : $(LLVM_OBJ_ROOT)/% : $(LLVM_SRC_ROOT)/%.in
	$(Echo) Regenerating $*
	$(Verb) cd $(LLVM_OBJ_ROOT) && $(ConfigStatusScript) $*
.PRECIOUS: $(FilesToConfigPATH)

# NOTE: This needs to remain as the last target definition in this file so
# that it gets executed last.
ifneq ($(BUILD_DIRS_ONLY),1)
all::
	$(Echo) '*****' Completed $(BuildMode)$(AssertMode) Build
ifeq ($(BuildMode),Debug)
	$(Echo) '*****' Note: Debug build can be 10 times slower than an
	$(Echo) '*****' optimized build. Use 'make ENABLE_OPTIMIZED=1' to
	$(Echo) '*****' make an optimized build. Alternatively you can
	$(Echo) '*****' configure with --enable-optimized.
endif
endif

check-llvm2cpp:
	$(Verb)$(MAKE) check TESTSUITE=Feature RUNLLVM2CPP=1

check-one:
	$(Verb)$(MAKE) -C test check-one TESTONE=$(TESTONE)

srpm: $(LLVM_OBJ_ROOT)/llvm.spec
	rpmbuild -bs $(LLVM_OBJ_ROOT)/llvm.spec

rpm: $(LLVM_OBJ_ROOT)/llvm.spec
	rpmbuild -bb --target $(TARGET_TRIPLE) $(LLVM_OBJ_ROOT)/llvm.spec

show-footprint:
	$(Verb) du -sk $(LibDir)
	$(Verb) du -sk $(ToolDir)
	$(Verb) du -sk $(ExmplDir)
	$(Verb) du -sk $(ObjDir)

build-for-llvm-top:
	$(Verb) if test ! -f ./config.status ; then \
	  ./configure --prefix="$(LLVM_TOP)/install" \
	    --with-llvm-gcc="$(LLVM_TOP)/llvm-gcc" ; \
	fi
	$(Verb) $(MAKE) tools-only

SVN = svn
SVN-UPDATE-OPTIONS =
AWK = awk
SUB-SVN-DIRS = $(AWK) '/\?\ \ \ \ \ \ / {print $$2}'   \
		| LANG=C xargs $(SVN) info 2>/dev/null \
		| $(AWK) '/Path:\ / {print $$2}'

update:
	$(SVN) $(SVN-UPDATE-OPTIONS) update $(LLVM_SRC_ROOT)
	@ $(SVN) status $(LLVM_SRC_ROOT) | $(SUB-SVN-DIRS) | xargs $(SVN) $(SVN-UPDATE-OPTIONS) update

happiness: update all check unittests

.PHONY: srpm rpm update happiness

# declare all targets at this level to be serial:

.NOTPARALLEL:

