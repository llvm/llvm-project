#===- test/Makefile ----------------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

LEVEL = ..
DIRS  =

all:: check-local

# Include other test rules
include Makefile.tests

#===------------------------------------------------------------------------===#
# DejaGNU testing support
#===------------------------------------------------------------------------===#

ifneq ($(GREP_OPTIONS),)
$(warning GREP_OPTIONS environment variable may interfere with test results)
endif

ifdef VERBOSE
LIT_ARGS := -v
else
LIT_ARGS := -s -v
endif

ifdef TESTSUITE
LIT_TESTSUITE := $(TESTSUITE)
else
LIT_TESTSUITE := .
endif

# Check what to run for -all.
LIT_ALL_TESTSUITES := $(LIT_TESTSUITE)

extra-site-cfgs::
.PHONY: extra-site-cfgs

ifneq ($(strip $(filter check-local-all,$(MAKECMDGOALS))),)
ifndef TESTSUITE
ifeq ($(shell test -f $(PROJ_OBJ_DIR)/../tools/clang/Makefile && echo OK), OK)
LIT_ALL_TESTSUITES += $(PROJ_OBJ_DIR)/../tools/clang/test

# Force creation of Clang's lit.site.cfg.
clang-site-cfg: FORCE
	$(MAKE) -C $(PROJ_OBJ_DIR)/../tools/clang/test lit.site.cfg Unit/lit.site.cfg
extra-site-cfgs:: clang-site-cfg
endif

ifeq ($(shell test -f $(PROJ_OBJ_DIR)/../tools/clang/tools/extra/Makefile && echo OK), OK)
LIT_ALL_TESTSUITES += $(PROJ_OBJ_DIR)/../tools/clang/tools/extra/test

# Force creation of Clang Tools' lit.site.cfg.
clang-tools-site-cfg: FORCE
	$(MAKE) -C $(PROJ_OBJ_DIR)/../tools/clang/tools/extra/test lit.site.cfg Unit/lit.site.cfg
extra-site-cfgs:: clang-tools-site-cfg
endif

ifeq ($(shell test -f $(PROJ_OBJ_DIR)/../tools/lld/Makefile && echo OK), OK)
LIT_ALL_TESTSUITES += $(PROJ_OBJ_DIR)/../tools/lld/test

# Force creation of lld's lit.site.cfg.
lld-site-cfg: FORCE
	$(MAKE) -C $(PROJ_OBJ_DIR)/../tools/lld/test lit.site.cfg Unit/lit.site.cfg
extra-site-cfgs:: lld-site-cfg
endif

ifeq ($(shell test -f $(PROJ_OBJ_DIR)/../tools/polly/Makefile && echo OK), OK)
LIT_ALL_TESTSUITES += $(PROJ_OBJ_DIR)/../tools/polly/test

# Force creation of Polly's lit.site.cfg.
polly-tools-site-cfg: FORCE
	$(MAKE) -C $(PROJ_OBJ_DIR)/../tools/polly/test lit.site.cfg
extra-site-cfgs:: polly-tools-site-cfg
endif
endif
endif

# ulimits like these are redundantly enforced by the buildbots, so
# just removing them here won't work.
# Solaris does not have the -m flag for ulimit
ifeq ($(HOST_OS),SunOS)
ULIMIT=ulimit -t 1200 ; ulimit -d 512000 ; ulimit -v 512000 ;
else # !SunOS
# Newer versions of python try to allocate an insane amount of address space for
# its thread-local storage, don't set a limit here.
# When -v is not used, then -s has to be used to limit the stack size.
# FIXME: Those limits should be enforced by lit instead of globally.
ULIMIT=ulimit -t 1200 ; ulimit -d 512000 ; ulimit -m 512000 ; ulimit -s 8192 ;
endif # SunOS

check-local:: lit.site.cfg Unit/lit.site.cfg
	( $(ULIMIT) \
	  $(PYTHON) $(LLVM_SRC_ROOT)/utils/lit/lit.py $(LIT_ARGS) $(LIT_TESTSUITE) )

# This is a legacy alias dating from when both DejaGNU and lit were in use.
check-local-lit:: check-local

check-local-all:: lit.site.cfg Unit/lit.site.cfg extra-site-cfgs
	( $(ULIMIT) \
	  $(PYTHON) $(LLVM_SRC_ROOT)/utils/lit/lit.py $(LIT_ARGS) $(LIT_ALL_TESTSUITES) )

clean::
	$(RM) -rf `find $(LLVM_OBJ_ROOT)/test -name Output -type d -print`

FORCE:

ifeq ($(DISABLE_ASSERTIONS),1)
ENABLE_ASSERTIONS=0
else
ENABLE_ASSERTIONS=1
endif

lit.site.cfg: FORCE
	@echo "Making LLVM 'lit.site.cfg' file..."
	@$(ECHOPATH) s=@LLVM_HOST_TRIPLE@=$(HOST_TRIPLE)=g > lit.tmp
	@$(ECHOPATH) s=@TARGET_TRIPLE@=$(TARGET_TRIPLE)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_TOOLS_DIR@=$(ToolDir)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_LIBRARY_DIR@=$(LibDir)=g >> lit.tmp
	@$(ECHOPATH) s=@SHLIBDIR@=$(SharedLibDir)=g >> lit.tmp
	@$(ECHOPATH) s=@SHLIBEXT@=$(SHLIBEXT)=g >> lit.tmp
	@$(ECHOPATH) s=@EXEEXT@=$(EXEEXT)=g >> lit.tmp
	@$(ECHOPATH) s=@PYTHON_EXECUTABLE@=$(PYTHON)=g >> lit.tmp
	@$(ECHOPATH) s=@GOLD_EXECUTABLE@=ld=g >> lit.tmp
	@$(ECHOPATH) s=@LD64_EXECUTABLE@=ld=g >> lit.tmp
	@$(ECHOPATH) s=@OCAMLFIND@=$(OCAMLFIND)=g >> lit.tmp
	@$(ECHOPATH) s!@OCAMLFLAGS@!$(addprefix -cclib ,$(LDFLAGS))!g >> lit.tmp
	@$(ECHOPATH) s=@HAVE_OCAMLOPT@=$(HAVE_OCAMLOPT)=g >> lit.tmp
	@$(ECHOPATH) s=@HAVE_OCAML_OUNIT@=$(HAVE_OCAML_OUNIT)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_INCLUDE_GO_TESTS@=ON=g >> lit.tmp
	@$(ECHOPATH) s=@GO_EXECUTABLE@=$(GO)=g >> lit.tmp
	@$(ECHOPATH) s!@HOST_CC@!$(CC)!g >> lit.tmp
	@$(ECHOPATH) s!@HOST_CXX@!$(CXX)!g >> lit.tmp
	@$(ECHOPATH) s!@HOST_LDFLAGS@!$(LDFLAGS)!g >> lit.tmp
	@$(ECHOPATH) s=@ENABLE_SHARED@=$(ENABLE_SHARED)=g >> lit.tmp
	@$(ECHOPATH) s=@ENABLE_ASSERTIONS@=$(ENABLE_ASSERTIONS)=g >> lit.tmp
	@$(ECHOPATH) s=@TARGETS_TO_BUILD@=$(TARGETS_TO_BUILD)=g >> lit.tmp
	@$(ECHOPATH) s=@LLVM_BINDINGS@=$(BINDINGS_TO_BUILD)=g >> lit.tmp
	@$(ECHOPATH) s=@HOST_OS@=$(HOST_OS)=g >> lit.tmp
	@$(ECHOPATH) s=@HOST_ARCH@=$(HOST_ARCH)=g >> lit.tmp
	@$(ECHOPATH) s=@HAVE_LIBZ@=$(HAVE_LIBZ)=g >> lit.tmp
	@$(ECHOPATH) s=@HAVE_DIA_SDK@=0=g >> lit.tmp
	@$(ECHOPATH) s=@ENABLE_EXAMPLES@=$(BUILD_EXAMPLES)=g >> lit.tmp
	@$(ECHOPATH) s=@ENABLE_TIMESTAMPS@=$(ENABLE_TIMESTAMPS)=g >> lit.tmp
	@sed -f lit.tmp $(PROJ_SRC_DIR)/lit.site.cfg.in > $@
	@-rm -f lit.tmp

Unit/lit.site.cfg: $(PROJ_OBJ_DIR)/Unit/.dir FORCE
	@echo "Making LLVM unittest 'lit.site.cfg' file..."
	@$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g > unit.tmp
	@$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> unit.tmp
	@$(ECHOPATH) s=@LLVM_TOOLS_DIR@=$(ToolDir)=g >> unit.tmp
	@$(ECHOPATH) s=@LLVM_BUILD_MODE@=$(BuildMode)=g >> unit.tmp
	@$(ECHOPATH) s=@ENABLE_SHARED@=$(ENABLE_SHARED)=g >> unit.tmp
	@$(ECHOPATH) s=@SHLIBDIR@=$(SharedLibDir)=g >> unit.tmp
	@$(ECHOPATH) s=@HOST_OS@=$(HOST_OS)=g >> unit.tmp
	@$(ECHOPATH) s=@HOST_ARCH@=$(HOST_ARCH)=g >> lit.tmp
	@sed -f unit.tmp $(PROJ_SRC_DIR)/Unit/lit.site.cfg.in > $@
	@-rm -f unit.tmp
