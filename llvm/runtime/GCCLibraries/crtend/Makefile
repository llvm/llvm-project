##===- runtime/GCCLibraries/crtend/Makefile ----------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This directory contains the C and C++ runtime libraries for the LLVM GCC
# front-ends.  See the README.txt file for more details.
#
# Since this archive has strange requirements, we use some custom rules for 
# building it.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
DONT_BUILD_RELINKED = 1
BYTECODE_LIBRARY = 1
LIBRARYNAME = crtend

# Note: We're using FAKE_SOURCES because the comp_*.c don't really exists. 
# However this makefile builds comp_*.bc and that's what we want in the library.
# The FAKE_SOURCES variable supports this kind of construction. It uses the
# FAKE_SOURCES to determine a list of things to build, but doesn't use 
# FAKE_SOURCES for dependencies, distribution, etc. It is assumed the makefile
# will know how to build the objects eventhough the sources don't exist.

FAKE_SOURCES := comp_main.c comp_genericeh.c comp_sjljeh.c

MainSrc      := crtend.c listend.ll
GenericEHSrc := Exception.cpp
SJLJEHSrc    := SJLJ-Exception.cpp

EXTRA_DIST   := $(MainSrc) $(GenericEHSrc) $(SJLJEHSrc) \
                comp_main.lst comp_genericeh.lst comp_sjljeh.lst

include $(LEVEL)/Makefile.common

MainObj      := $(ObjDir)/crtend.bc $(ObjDir)/listend.bc
GenericEHObj := $(ObjDir)/Exception.bc
SJLJEHObj    := $(ObjDir)/SJLJ-Exception.bc

# __main and ctor/dtor support component
$(ObjDir)/comp_main.bc: $(MainObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_main.lst \
	$(MainObj) -o $@

# Generic exception handling support runtime.
$(ObjDir)/comp_genericeh.bc: $(GenericEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_genericeh.lst \
	$(GenericEHObj) -o $@ 

# setjmp/longjmp exception handling support runtime.
$(ObjDir)/comp_sjljeh.bc: $(SJLJEHObj)
	$(Echo) Linking $(notdir $@) component...
	$(Verb) $(GCCLD) -link-as-library \
	-internalize-public-api-file=$(BUILD_SRC_DIR)/comp_sjljeh.lst \
	$(SJLJEHObj) -o $@
