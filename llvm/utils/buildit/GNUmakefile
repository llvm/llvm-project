# LLVM LOCAL file build machinery
# LLVM Compiler Makefile for use by buildit.  
#
# This makefile is intended only for use with B&I buildit. For "normal" builds
# use the conventional top-level makefile.
#
# You can specify TARGETS=ppc (or i386) on the buildit command line to limit the
# build to just one target. The default is for ppc and i386. The compiler
# targetted at this host gets built anyway, but not installed unless it's listed
# in TARGETS.

# Include the set of standard Apple makefile definitions.
ifndef CoreOSMakefiles
CoreOSMakefiles = $(MAKEFILEPATH)/CoreOS
endif
include $(CoreOSMakefiles)/Standard/Standard.make

# Enable Apple extensions to (gnu)make.
USE_APPLE_PB_SUPPORT = all

RC_ARCHS := ppc i386
HOSTS = $(RC_ARCHS)
targets = echo $(RC_ARCHS)
TARGETS := $(shell $(targets))

SRCROOT = .

SRC = $(shell cd $(SRCROOT) && pwd | sed s,/private,,)
OBJROOT = $(SRC)/obj
SYMROOT = $(OBJROOT)/../sym
DSTROOT = $(OBJROOT)/../dst

#######################################################################

PREFIX = /usr/local

# Unless assertions are forced on in the GMAKE command line, disable them.
ifdef ENABLE_ASSERTIONS
LLVM_ASSERTIONS := yes
else
LLVM_ASSERTIONS := no
endif

# Default is optimized build.
ifeq ($(LLVM_DEBUG),1)
LLVM_OPTIMIZED := no
else
LLVM_OPTIMIZED := yes
endif

ifndef RC_ProjectSourceVersion
RC_ProjectSourceVersion = 9999
endif

ifndef RC_ProjectSourceSubversion
RC_ProjectSourceSubversion = 0
endif

# NOTE : Always put version numbers at the end because they are optional.
install: $(OBJROOT) $(SYMROOT) $(DSTROOT)
	cd $(OBJROOT) && \
	  $(SRC)/build_llvm "$(RC_ARCHS)" "$(TARGETS)" \
	    $(SRC) $(PREFIX) $(DSTROOT) $(SYMROOT) \
	    $(LLVM_ASSERTIONS) $(LLVM_OPTIMIZED) \
	    $(RC_ProjectSourceVersion) $(RC_ProjectSourceSubversion) 


# installhdrs does nothing, because the headers aren't useful until
# the compiler is installed.
installhdrs:

# We build and install in one shell script.
build: 

installsrc:
	@echo
	@echo ++++++++++++++++++++++
	@echo + Installing sources +
	@echo ++++++++++++++++++++++
	@echo
	if [ $(SRCROOT) != . ]; then \
	  $(PAX) -rw . $(SRCROOT); \
	fi
	find -d "$(SRCROOT)" \( -type d -a -name .svn -o \
	                        -type f -a -name .DS_Store -o \
				-name \*~ -o -name .\#\* \) \
	  -exec rm -rf {} \;

#######################################################################

clean:
	@echo
	@echo ++++++++++++
	@echo + Cleaning +
	@echo ++++++++++++
	@echo
	@if [ -d $(OBJROOT) -a "$(OBJROOT)" != / ]; then \
	  echo '*** DELETING ' $(OBJROOT); \
	  rm -rf $(OBJROOT); \
	fi
	@if [ -d $(SYMROOT) -a "$(SYMROOT)" != / ]; then \
	  echo '*** DELETING ' $(SYMROOT); \
	  rm -rf $(SYMROOT); \
	fi
	@if [ -d $(DSTROOT) -a "$(DSTROOT)" != / ]; then \
	  echo '*** DELETING ' $(DSTROOT); \
	  rm -rf $(DSTROOT); \
	fi

#######################################################################

$(OBJROOT) $(SYMROOT) $(DSTROOT):
	mkdir -p $@

.PHONY: install installsrc clean
