##===- runtime/libprofile/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
include $(LEVEL)/Makefile.config

ifneq ($(strip $(LLVMCC)),)
BYTECODE_LIBRARY = 1
endif
LIBRARYNAME = profile_rt
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1
EXTRA_DIST = libprofile.exports
EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/libprofile.exports

# Build and install this archive.                                                                                                                  
BUILD_ARCHIVE = 1
override NO_INSTALL_ARCHIVES =

include $(LEVEL)/Makefile.common

ifeq ($(HOST_OS),Darwin)
    # Special hack to allow libprofile_rt to have an offset version number.
    PROFILE_RT_LIBRARY_VERSION := $(LLVM_SUBMIT_VERSION)

    # Set dylib internal version number to llvmCore submission number.
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
                        -Wl,$(PROFILE_RT_LIBRARY_VERSION).$(LLVM_SUBMIT_SUBVERSION) \
                        -Wl,-compatibility_version -Wl,1
    endif
    # Extra options to override libtool defaults.
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -Wl,-dead_strip \
                         -Wl,-seg1addr -Wl,0xE0000000 

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on
    # command line.
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions) \
                            -Wl,-install_name \
                            -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif

    # If we're doing an Apple-style build, add the LTO object path.
    ifeq ($(RC_BUILDIT),YES)
       TempFile           := $(shell mkdir -p ${OBJROOT}/dSYMs ; mktemp ${OBJROOT}/dSYMs/profile_rt-lto.XXXXXX)
       LLVMLibsOptions    := $(LLVMLibsOptions) \
                             -Wl,-object_path_lto -Wl,$(TempFile)
    endif
endif
