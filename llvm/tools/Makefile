##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ..

include $(LEVEL)/Makefile.config

# Build clang if present.

ifneq ($(CLANG_SRC_ROOT),)
  OPTIONAL_PARALLEL_DIRS := $(CLANG_SRC_ROOT)
else
  OPTIONAL_PARALLEL_DIRS := clang
endif

# Build LLDB if present. Note LLDB must be built last as it depends on
# the wider LLVM infrastructure (including Clang).
OPTIONAL_DIRS := lldb

# NOTE: The tools are organized into five groups of four consisting of one
# large and three small executables. This is done to minimize memory load
# in parallel builds.  Please retain this ordering.
DIRS := llvm-config
PARALLEL_DIRS := opt llvm-as llvm-dis llc llvm-ar llvm-nm llvm-link \
                 lli llvm-extract llvm-mc bugpoint llvm-bcanalyzer llvm-diff \
                 llvm-objdump llvm-readobj llvm-rtdyld \
                 llvm-dwarfdump llvm-cov llvm-size llvm-stress llvm-mcmarkup \
                 llvm-profdata llvm-symbolizer obj2yaml yaml2obj llvm-c-test \
                 llvm-cxxdump verify-uselistorder dsymutil llvm-pdbdump \
                 llvm-split sancov llvm-dwp

# If Intel JIT Events support is configured, build an extra tool to test it.
ifeq ($(USE_INTEL_JITEVENTS), 1)
  PARALLEL_DIRS += llvm-jitlistener
endif

# Let users override the set of tools to build from the command line.
ifdef ONLY_TOOLS
  OPTIONAL_PARALLEL_DIRS :=
  OPTIONAL_DIRS := $(findstring lldb,$(ONLY_TOOLS))
  PARALLEL_DIRS := $(filter-out lldb,$(ONLY_TOOLS))
endif

# These libraries build as dynamic libraries (.dylib /.so), they can only be
# built if ENABLE_PIC is set.
ifndef ONLY_TOOLS
ifeq ($(ENABLE_PIC),1)
  # gold only builds if binutils is around.  It requires "lto" to build before
  # it so it is added to DIRS. llvm-lto also requires lto
  DIRS += lto llvm-lto
  ifdef BINUTILS_INCDIR
    DIRS += gold
  endif

  PARALLEL_DIRS += bugpoint-passes
endif

ifdef LLVM_HAS_POLLY
  PARALLEL_DIRS += polly
endif
endif

# On Win32, loadable modules can be built with ENABLE_SHARED.
ifneq ($(ENABLE_SHARED),1)
  ifneq (,$(filter $(HOST_OS), Cygwin MingW))
    PARALLEL_DIRS := $(filter-out bugpoint-passes, \
                        $(PARALLEL_DIRS))
  endif
endif

ifneq (,$(filter go,$(BINDINGS_TO_BUILD)))
  PARALLEL_DIRS += llvm-go
endif

include $(LEVEL)/Makefile.common
