# common-defs.mk #

#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# --------------------------------------------------------------------------------------------------
# This file contains really common definitions used by multiple makefiles. Modify it carefully!
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Some tricky variables.
# --------------------------------------------------------------------------------------------------
empty :=
space := $(empty) $(empty)
comma := ,
ifeq "$(date)" ""
    $(error Caller should specify "date" variable.)
endif

# --------------------------------------------------------------------------------------------------
# Helper finctions.
# --------------------------------------------------------------------------------------------------

# Synopsis:
#     $(call say,text-to-print-to-the-screen)
# Description:
#     The function prints its argument to the screen. In case of older makes it is analog of
#     $(warning), starting from make 3.81 is is analog of $(info).
#
say = $(warning $(1))
ifneq "$(filter 3.81,$(MAKE_VERSION))" ""
    say = $(info $(1))
endif

# Synopsis:
#     architecture = $(call legal_arch,32)
# Description:
#     The function return printable name of specified architecture, IA-32 architecture or Intel(R) 64.
#
legal_arch = $(if $(filter 32,$(1)),IA-32,$(if $(filter 32e,$(1)),Intel(R) 64,$(if $(filter l1,$(1)),L1OM,$(if $(filter arm,$(1)),ARM,$(if $(filter ppc64,$(1)),PPC64,$(error Bad architecture specified: $(1)))))))

# Synopsis:
#     var_name = $(call check_variable,var,list)
# Description:
#     The function verifies the value of var varibale. If it is empty, the first word from the list
#     is assigned to var as default value. Otherwise the var value must match one of words in the
#     list, or error is issued.
# Example:
#     LINK_TYPE = $(call check_variable,LINK_TYPE,static dynamic)
#
check_variable = $(call _check_variable_words,$(1))$(call _check_variable_value,$(1),$(2))

# Synopsis:
#     $(call _check_variable_words,var)
# Description:
#     Checks that variable var is empty or single word. In case of multiple words an error is
#     issued. It is helper function for check_variable.
#
_check_variable_words = $(if $(filter 0 1,$(words $($(1)))),,\
    $(error Illegal value of $(1): "$($(1))"))

# Synopsis:
#     $(call _check_variable_value,var)
# Description:
#     If variable var is empty, the function returns the first word from the list. If variable is
#     not empty and match one of words in the list, variable's value returned. Otherwise, error is
#     issued. It is helper function for check_variable.
#
_check_variable_value = $(if $($(1)),$(if $(filter $(2),$($(1))),$($(1)),\
    $(error Illegal value of $(1): "$($(1))")),$(firstword $(2)))

# Synopsis:
#     $(call debug,var)
# Description:
#     If LIBOMP_MAKE_DEBUG is not empty, var name and value printed. Use this for debug purpose.
#
ifeq "$(LIBOMP_MAKE_DEBUG)" ""
    debug =
else
    debug = $(call say,debug: $(1)="$($(1))")
endif

# Synopsis:
#     $(call header,target)
# Description:
#     Returns a string to print to show build progress.
#
header = ----- $(marker) --- $(1) -----

# --------------------------------------------------------------------------------------------------
# Global make settings.
# --------------------------------------------------------------------------------------------------

# Non-empty CDPATH may lead to problems on some platforms: simple "cd dir" (where "dir" is an
# existing directory in current one) fails. Clearing CDPATH solves the problem.
CDPATH =
.SUFFIXES :            # Clean default list of suffixes.
.DELETE_ON_ERROR :     # Delete target file in case of error.

$(call say,$(call header,making $(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)))

# --------------------------------------------------------------------------------------------------
# Check clean and clobber goals.
# --------------------------------------------------------------------------------------------------

# "clean" goal must be specified alone, otherwise we have troubles with dependency files.
clean := $(filter clean%,$(MAKECMDGOALS))
ifneq "$(clean)" ""                                    # "clean" goal is present in command line.
    ifneq "$(filter-out clean%,$(MAKECMDGOALS))" ""    # there are non-clean goals.
        $(error "clean" goals must not be mixed with other goals)
    endif
endif
# Issue error on "clobber" target.
ifneq "$(filter clobber,$(MAKECMDGOALS))" ""
    $(error There is no clobber goal in makefile)
endif

# --------------------------------------------------------------------------------------------------
# Mandatory variables passed from build.pl.
# --------------------------------------------------------------------------------------------------

os       := $(call check_variable,os,lin lrb mac win)
arch     := $(call check_variable,arch,32 32e 64 arm ppc64)
platform := $(os)_$(arch)
platform := $(call check_variable,platform,lin_32 lin_32e lin_64 lin_arm lrb_32e mac_32 mac_32e win_32 win_32e win_64 lin_ppc64)
# oa-opts means "os and arch options". They are passed to almost all perl scripts.
oa-opts  := --os=$(os) --arch=$(arch)

# --------------------------------------------------------------------------------------------------
# Directories.
# --------------------------------------------------------------------------------------------------

ifeq "$(LIBOMP_WORK)" ""
    $(error Internal error: LIBOMP_WORK variable must be set in makefile.mk)
endif
tools_dir = $(LIBOMP_WORK)tools/
# We do not define src/ and other directories here because they depends on target (RTL, DSL, tools).

# --------------------------------------------------------------------------------------------------
# File suffixes.
# --------------------------------------------------------------------------------------------------

ifeq "$(os)" "win" # win
    asm = .asm
    obj = .obj
    lib = .lib
    dll = .dll
    exe = .exe
    cat = $(dll)
else # lin, lrb or mac
    asm = .s
    obj = .o
    lib = .a
    ifeq "$(os)" "mac"
        dll = .dylib
    else
        dll = .so
    endif
    exe = $(empty)
    cat = .cat
endif

# --------------------------------------------------------------------------------------------------
# File manipulation and misc commands.
# --------------------------------------------------------------------------------------------------

target = @echo "$(call header,$@)"
ifeq "$(os)" "win"
    cp    = cp -f
    rm    = rm -f
    mkdir = mkdir -p
    touch = touch
    perl  = perl
    slash = \\
else # lin, lrb or mac
    cp    = cp -f
    rm    = rm -f
    mkdir = mkdir -p
    touch = touch
    perl  = perl
    slash = /
endif

# --------------------------------------------------------------------------------------------------
# Common non-configuration options.
# --------------------------------------------------------------------------------------------------
# They may affect build process but does not affect result.

# If TEST_DEPS is "off", test deps is still performed, but its result is ignored.
TEST_DEPS  := $(call check_variable,TEST_DEPS,on off)
# The same for test touch.
TEST_TOUCH := $(call check_variable,TEST_TOUCH,on off)
td-i = $(if $(filter off,$(TEST_DEPS)),-)
tt-i = $(if $(filter off,$(TEST_TOUCH)),-)

# --------------------------------------------------------------------------------------------------
# Common targets.
# --------------------------------------------------------------------------------------------------

# All common targets are defined as phony. It allows "buil.pl --all test-xxx".
# Makefile can define actions for a particiular test or leave it no-op.

# all, the default target, should be the first one.
.PHONY : all
all :

.PHONY : common clean clean-common fat inc l10n lib

.PHONY : force-tests          tests
.PHONY : force-test-touch     test-touch
.PHONY : force-test-relo      test-relo
.PHONY : force-test-execstack test-execstack
.PHONY : force-test-instr     test-instr
.PHONY : force-test-deps      test-deps

tests = touch relo execstack instr deps
tests       : $(addprefix test-,$(tests))
force-tests : $(addprefix force-test-,$(tests))

# end of file #
