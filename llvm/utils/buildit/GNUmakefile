# LLVM LOCAL file build machinery
# LLVM Compiler Makefile for use by buildit.  
#
# This makefile is intended only for use with B&I buildit. For "normal" builds
# use the conventional top-level makefile.
#
# You can specify TARGETS=ppc (or i386) on the buildit command line to limit the
# build to just one target. The default is for ppc and i386. The compiler
# targeted at this host gets built anyway, but not installed unless it's listed
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
ifndef ENABLE_ASSERTIONS
ENABLE_ASSERTIONS := no
endif

# Default is optimized build.
ifeq ($(LLVM_DEBUG),1)
LLVM_OPTIMIZED := no
else
LLVM_OPTIMIZED := yes
endif

# Default to do a native build, not a cross-build for an ARM host or simulator.
ARM_HOSTED_BUILD := no
IOS_SIM_BUILD := no

ifndef RC_ProjectSourceVersion
RC_ProjectSourceVersion = 9999
endif

ifndef RC_ProjectSourceSubversion
RC_ProjectSourceSubversion = 0
endif

# NOTE : Always put version numbers at the end because they are optional.
install: $(OBJROOT) $(SYMROOT) $(DSTROOT)
	cd $(OBJROOT) && \
	  $(SRC)/utils/buildit/build_llvm "$(RC_ARCHS)" "$(TARGETS)" \
	    $(SRC) $(PREFIX) $(DSTROOT) $(SYMROOT) \
	    $(ENABLE_ASSERTIONS) $(LLVM_OPTIMIZED) \
	    $(ARM_HOSTED_BUILD) $(IOS_SIM_BUILD) \
	    $(RC_ProjectSourceVersion) $(RC_ProjectSourceSubversion) 

EmbeddedHosted:
	$(MAKE) ARM_HOSTED_BUILD=yes PREFIX=/usr/local install

# When building for the iOS simulator, MACOSX_DEPLOYMENT_TARGET is not set
# by default, but it needs to be set when building tools that run on the host
# (e.g., tblgen), so set it here.
EmbeddedSim:
	export MACOSX_DEPLOYMENT_TARGET=`sw_vers -productVersion`; \
	$(MAKE) IOS_SIM_BUILD=yes PREFIX=$(SDKROOT)/usr/local install

Embedded:
	ARM_PLATFORM=`xcodebuild -version -sdk iphoneos PlatformPath` && \
	$(MAKE) DSTROOT=$(DSTROOT)$$ARM_PLATFORM/Developer install

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
	rm -rf "$(SRCROOT)/test"

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

.PHONY: install installsrc clean EmbeddedHosted EmbeddedSim Embedded
