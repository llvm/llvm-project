##===- tools/lto/Makefile ----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
LIBRARYNAME := LTO
LINK_COMPONENTS := all-targets ipo scalaropts linker bitreader bitwriter \
                   mcdisassembler vectorize
LINK_LIBS_IN_SHARED := 1
SHARED_LIBRARY := 1

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/lto.exports

include $(LEVEL)/Makefile.common

ifdef LLVM_VERSION_INFO
CXX.Flags += -DLLVM_VERSION_INFO='"$(LLVM_VERSION_INFO)"'
endif

ifeq ($(HOST_OS),Darwin)
    # Special hack to allow libLTO to have an offset version number.
    ifdef LLVM_LTO_VERSION_OFFSET
        LTO_LIBRARY_VERSION := $(shell expr $(LLVM_SUBMIT_VERSION) + \
                                            $(LLVM_LTO_VERSION_OFFSET))
    else
        LTO_LIBRARY_VERSION := $(LLVM_SUBMIT_VERSION)
    endif

    # set dylib internal version number to llvmCore submission number
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
                        -Wl,$(LTO_LIBRARY_VERSION).$(LLVM_SUBMIT_SUBVERSION) \
                        -Wl,-compatibility_version -Wl,1
    endif
    # extra options to override libtool defaults 
    LLVMLibsOptions    := $(LLVMLibsOptions)  \
                         -Wl,-dead_strip \
                         -Wl,-seg1addr -Wl,0xE0000000 

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions    := $(LLVMLibsOptions)  \
                            -Wl,-install_name \
                            -Wl,"@executable_path/../lib/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif

    # If we're doing an Apple-style build, add the LTO object path.
    ifeq ($(RC_BUILDIT),YES)
       TempFile        := $(shell mkdir -p ${OBJROOT}/dSYMs ; mktemp ${OBJROOT}/dSYMs/llvm-lto.XXXXXX)
       LLVMLibsOptions := $(LLVMLibsOptions) \
                          -Wl,-object_path_lto -Wl,$(TempFile)
    endif
endif
