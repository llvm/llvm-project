##===- tools/llvm-config/Makefile---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := llvm-config
USEDLIBS := LLVMSupport.a

# We generate sources in the build directory, make sure it is in the include
# paths.
INCLUDE_BUILD_DIR := 1

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS := 1

# Note that we have to use lazy expansion here.
BUILDVARIABLES_SRCPATH = $(PROJ_SRC_ROOT)/tools/$(TOOLNAME)/BuildVariables.inc.in
BUILDVARIABLES_OBJPATH = $(ObjDir)/BuildVariables.inc
BUILT_SOURCES = $(BUILDVARIABLES_OBJPATH)

include $(LEVEL)/Makefile.common

# Combine preprocessor flags (except for -I) and CXX flags.
SUB_CPPFLAGS := ${CPP.BaseFlags}
SUB_CFLAGS   := ${CPP.BaseFlags} ${C.Flags}
SUB_CXXFLAGS := ${CPP.BaseFlags} ${CXX.Flags}

# Override LIBS with TARGET's LIBS for cross compilation.
# FIXME: Host's llvm-config is not generated. It's for target's.
ifneq ($(TARGET_LIBS), )
  LLVM_SYSTEM_LIBS := $(TARGET_LIBS)
else
  LLVM_SYSTEM_LIBS := $(LIBS)
endif

ifneq ($(REQUIRES_RTTI), 1)
  LLVM_HAS_RTTI := NO
else
  LLVM_HAS_RTTI := YES
endif

# This is blank for now.  We need to be careful about adding stuff here:
# LDFLAGS tend not to be portable, and we don't currently require the
# user to use libtool when linking against LLVM.
SUB_LDFLAGS :=

$(ObjDir)/BuildVariables.inc: $(BUILDVARIABLES_SRCPATH) Makefile $(ObjDir)/.dir
	$(Echo) "Building llvm-config BuildVariables.inc file."
	$(Verb) $(ECHO) 's/@LLVM_SRC_ROOT@/$(subst /,\/,$(LLVM_SRC_ROOT))/' \
	  > temp.sed
	$(Verb) $(ECHO) 's/@LLVM_OBJ_ROOT@/$(subst /,\/,$(LLVM_OBJ_ROOT))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_CPPFLAGS@/$(subst /,\/,$(SUB_CPPFLAGS))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_CFLAGS@/$(subst /,\/,$(SUB_CFLAGS))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_CXXFLAGS@/$(subst /,\/,$(SUB_CXXFLAGS))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_LDFLAGS@/$(subst /,\/,$(SUB_LDFLAGS))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_BUILDMODE@/$(subst /,\/,$(BuildMode))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_LIBDIR_SUFFIX@//' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_SYSTEM_LIBS@/$(subst /,\/,$(LLVM_SYSTEM_LIBS))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_TARGETS_BUILT@/$(subst /,\/,$(TARGETS_TO_BUILD))/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_BUILD_SYSTEM@/autoconf/' \
	  >> temp.sed
	$(Verb) $(ECHO) 's/@LLVM_HAS_RTTI@/$(LLVM_HAS_RTTI)/' \
	  >> temp.sed
	$(Verb) $(SED) -f temp.sed < $< > $@
	$(Verb) $(RM) temp.sed

# When cross-compiling, install a version of llvm-config that runs on the host.
ifeq ($(LLVM_CROSS_COMPILING),1)
install:: $(DESTDIR)$(PROJ_bindir)
	$(Echo) Installing llvm-config-host
	$(Verb) $(ProgInstall) $(BuildLLVMToolDir)/llvm-config \
	  $(DESTDIR)$(PROJ_bindir)/$(program_prefix)llvm-config-host
endif
