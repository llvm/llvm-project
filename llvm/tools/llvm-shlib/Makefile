##===- tools/shlib/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..

LIBRARYNAME = LLVM-$(LLVMVersion)

NO_BUILD_ARCHIVE := 1
LINK_LIBS_IN_SHARED := 1
SHARED_LIBRARY := 1

include $(LEVEL)/Makefile.config

ifeq ($(HOST_OS), $(filter $(HOST_OS), Cygwin MingW))
  EXPORTED_SYMBOL_FILE = $(ObjDir)/$(LIBRARYNAME).exports

  ifeq (1,$(ENABLE_EMBED_STDCXX))
    # It is needed to force static-stdc++.a linked.
    SHLIB_FRAG_NAMES += stdc++.a.o
  endif

endif

include $(LEVEL)/Makefile.common

# Include all archives in libLLVM.(so|dylib) except the ones that have
# their own dynamic libraries and TableGen.
Archives := $(wildcard $(LibDir)/libLLVM*.a)
SharedLibraries := $(wildcard $(LibDir)/libLLVM*$(SHLIBEXT))
ExcludeFromLibLlvm := $(basename $(SharedLibraries)).a %/libLLVMTableGen.a
IncludeInLibLlvm := $(filter-out $(ExcludeFromLibLlvm), $(Archives))
LLVMLibsOptions := $(IncludeInLibLlvm:$(LibDir)/lib%.a=-l%)
LLVMLibsPaths   := $(IncludeInLibLlvm)

$(LibName.SO): $(LLVMLibsPaths)

ifeq ($(HOST_OS),Darwin)
    # set dylib internal version number to llvmCore submission number
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions := $(LLVMLibsOptions) -Wl,-current_version \
                        -Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION) \
                        -Wl,-compatibility_version -Wl,1
    endif
    # Include everything from the .a's into the shared library.
    LLVMLibsOptions    := $(LLVMLibsOptions) -all_load
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
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux FreeBSD OpenBSD GNU))
    # Include everything from the .a's into the shared library.
    LLVMLibsOptions := -Wl,--whole-archive $(LLVMLibsOptions) \
                       -Wl,--no-whole-archive
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux FreeBSD GNU))
    # Add soname to the library.
    LLVMLibsOptions += -Wl,--soname,lib$(LIBRARYNAME)$(SHLIBEXT)
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux GNU))
    # Don't allow unresolved symbols.
    LLVMLibsOptions += -Wl,--no-undefined
endif

ifeq ($(HOST_OS),SunOS)
    # add -z allextract ahead of other libraries on Solaris
    LLVMLibsOptions := -Wl,-z -Wl,allextract $(LLVMLibsOptions)
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Cygwin MingW))

SHLIB_STUBS := $(addprefix $(ObjDir)/, $(SHLIB_FRAG_NAMES))
SHLIB_FRAGS := $(patsubst %.a.o, $(ObjDir)/%.syms.txt, $(LIBRARYNAME).a.o $(SHLIB_FRAG_NAMES))
LLVMLibsOptions := $(SHLIB_STUBS) $(LLVMLibsOptions)

$(LibName.SO): $(SHLIB_STUBS)

%.syms.txt: %.a.o
	$(Echo) Collecting global symbols of $(notdir $*)
	$(Verb) $(NM_PATH) -g $< > $@

$(ObjDir)/$(LIBRARYNAME).exports: $(SHLIB_FRAGS) $(ObjDir)/.dir
	$(Echo) Generating exports for $(LIBRARYNAME)
	$(Verb) ($(SED) -n \
			-e "s/^.* T _\([^.][^.]*\)$$/\1/p" \
			-e "s/^.* [BDR] _\([^.][^.]*\)$$/\1 DATA/p" \
			$(SHLIB_FRAGS) \
		 | sort -u) > $@

$(ObjDir)/$(LIBRARYNAME).a.o: $(LLVMLibsPaths) $(ObjDir)/.dir
	$(Echo) Linking all LLVMLibs together for $(LIBRARYNAME)
	$(Verb) $(Link) -nostartfiles -Wl,-r -nodefaultlibs -o $@ \
			-Wl,--whole-archive $(LLVMLibsPaths) \
			-Wl,--no-whole-archive

$(ObjDir)/stdc++.a.o: $(ObjDir)/.dir
	$(Echo) Linking all libs together for static libstdc++.a
	$(Verb) $(Link) -nostartfiles -Wl,-r -nodefaultlibs -o $@ \
			-Wl,--whole-archive -lstdc++ \
			-Wl,--no-whole-archive
# FIXME: workaround to invalidate -lstdc++
	$(Echo) Making dummy -lstdc++ to lib
	$(Verb) $(AR) rc $(ToolDir)/libstdc++.dll.a
# FIXME: Is install-local needed?

clean-local::
	$(Verb) $(RM) -f $(ToolDir)/libstdc++.dll.a

endif
