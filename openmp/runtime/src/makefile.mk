# makefile.mk #
# $Revision: 42820 $
# $Date: 2013-11-13 16:53:44 -0600 (Wed, 13 Nov 2013) $

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

# Check and normalize LIBOMP_WORK.
# This piece of code is common, but it cannot be moved to common file.
ifeq "$(LIBOMP_WORK)" ""
    $(error LIBOMP_WORK environment variable must be set)
endif
ifneq "$(words $(LIBOMP_WORK))" "1"
    $(error LIBOMP_WORK must not contain spaces)
endif
override LIBOMP_WORK := $(subst \,/,$(LIBOMP_WORK))
ifeq "$(filter %/,$(LIBOMP_WORK))" ""
    override LIBOMP_WORK := $(LIBOMP_WORK)/
endif

# Include definitions common for RTL and DSL.
include $(LIBOMP_WORK)src/defs.mk

src_dir      = $(LIBOMP_WORK)src/
inc_dir      = $(LIBOMP_WORK)src/include/$(OMP_VERSION)/

# --------------------------------------------------------------------------------------------------
# Configuration options.
# --------------------------------------------------------------------------------------------------

# Build compiler
BUILD_COMPILER := $(call check_variable,BUILD_COMPILER,icc gcc clang icl icl.exe)
# Distribution type: com (commercial) or oss (open-source)
DISTRIBUTION  := $(call check_variable,DISTRIBUTION,com oss)

ifeq "$(c)" ""
    c = $(BUILD_COMPILER)
    ifeq "$(os)" "win"
        c = icl.exe
    endif
endif
ifeq "$(dist)" ""
    dist = $(DISTRIBUTION)
endif
ifeq "$(dist)" ""
    dist = com
endif

# Compile all C files as C++ source.
CPLUSPLUS    := $(call check_variable,CPLUSPLUS,on)
# Turn on instrumentation for code coverage.
COVERAGE     := $(call check_variable,COVERAGE,off on)
# Instruct compiler to emit debug information.
DEBUG_INFO   := $(call check_variable,DEBUG_INFO,on off)
# Turn on debug support in library code, assertions and traces.
DIAG         := $(call check_variable,DIAG,on off)
LIB_TYPE     := $(call check_variable,LIB_TYPE,norm prof stub)
# Type of library: dynamic or static linking.
LINK_TYPE    := $(call check_variable,LINK_TYPE,dyna stat)
# Supported OpenMP version, 2.5 or 3.0.
OMP_VERSION  := $(call check_variable,OMP_VERSION,40 30 25)
# Generate optimized code.
OPTIMIZATION := $(call check_variable,OPTIMIZATION,off on)
# Target compiler.
TARGET_COMPILER := $(call check_variable,TARGET_COMPILER,12 11)
# Library version: 4 -- legacy, 5 -- compat.
VERSION      := $(call check_variable,VERSION,5 4)
# quad precision floating point
HAVE_QUAD     = 1

VPATH += $(src_dir)
VPATH += $(src_dir)i18n/
VPATH += $(inc_dir)
VPATH += $(src_dir)thirdparty/ittnotify/


# Define config.
define curr_config
    CPLUSPLUS=$(CPLUSPLUS)
    COVERAGE=$(COVERAGE)
    DEBUG_INFO=$(DEBUG_INFO)
    DIAG=$(DIAG)
    LIB_TYPE=$(LIB_TYPE)
    LINK_TYPE=$(LINK_TYPE)
    OMP_VERSION=$(OMP_VERSION)
    OPTIMIZATION=$(OPTIMIZATION)
    TARGET_COMPILER=$(TARGET_COMPILER)
    VERSION=$(VERSION)
    CPPFLAGS=$(subst $(space),_,$(CPPFLAGS))
    CFLAGS=$(subst $(space),_,$(CFLAGS))
    CXXFLAGS=$(subst $(space),_,$(CXXFLAGS))
    FFLAGS=$(subst $(space),_,$(FFLAGS))
    LDFLAGS=$(subst $(space),_,$(LDFLAGS))
endef
# And check it.
include $(tools_dir)src/common-checks.mk

# Function to convert LIB_TYPE to printable one.
legal_type = $(if $(filter norm,$(LIB_TYPE)),Performance,$(if $(filter prof,$(LIB_TYPE)),Profiling,Stub))

# Check the OS X version (we need it to decide which tool use for objects accumulation)
ifeq "$(os)" "mac"
    mac_os_new := $(shell /bin/sh -c 'if [[ `sw_vers -productVersion` > 10.6 ]]; then echo "1"; else echo "0"; fi')
endif

# Form target directory name for MIC platforms
ifeq "$(MIC_ARCH)" "knc"
    mic-postf1  = .knc
endif
ifeq "$(MIC_OS)" "lin"
    mic-postfix = $(mic-postf1).lin
else
    mic-postfix = $(mic-postf1)
endif


# --------------------------------------------------------------------------------------------------
# Dev tools and general options (like -fpic, -O2 or -g).
# --------------------------------------------------------------------------------------------------
include $(tools_dir)src/common-tools.mk

# --------------------------------------------------------------------------------------------------
# Project-specific tools options.
# --------------------------------------------------------------------------------------------------

# --- Assembler options ---

ifeq "$(os)" "win"
    ifeq "$(arch)" "32"
        as-flags += -coff
        as-flags += -D_M_IA32
    endif
    ifeq "$(arch)" "32e"
        as-flags  += -D_M_AMD64
    endif
    ifeq "$(arch)" "64"
    endif
endif

# --- C/C++ options ---

# Enable _Quad type.
ifneq "$(filter icc icl icl.exe,$(c))" ""
    c-flags   += -Qoption,cpp,--extended_float_types
    cxx-flags += -Qoption,cpp,--extended_float_types
endif

ifeq "$(c)" "gcc"
    ifeq "$(arch)" "32"
        c-flags += -m32 -msse
        cxx-flags += -m32 -msse
        fort-flags += -m32 -msse
        ld-flags += -m32 -msse
        as-flags += -m32 -msse
    endif
endif

ifeq "$(c)" "clang"
    c-flags += -Wno-unused-value -Wno-switch -Wno-deprecated-register
    cxx-flags += -Wno-unused-value -Wno-switch -Wno-deprecated-register
    ifeq "$(arch)" "32"
        c-flags += -m32 -msse
        cxx-flags += -m32 -msse
        fort-flags += -m32 -msse
        ld-flags += -m32 -msse
        as-flags += -m32 -msse
    endif
    HAVE_QUAD = 0
endif

ifeq "$(LINK_TYPE)" "dyna"
# debug-info
    ifeq "$(os)" "win"
        c-flags    += -Zi
        cxx-flags  += -Zi
        fort-flags += -Zi
    else
      ifneq "$(os)" "mac"
        c-flags    += -g
        cxx-flags  += -g
        fort-flags += -g
        ld-flags   += -g
      endif
    endif
endif

# Enable 80-bit "long double".
# ??? In original makefile, it was enabled for all files on win_32 and win_64, and only for one
# file kmp_atomic.c on win_32e.
ifeq "$(os)" "win"
    c-flags   += -Qlong_double
    cxx-flags += -Qlong_double
endif

# Enable saving compiler options and version in object files and libraries.
ifeq "$(filter gcc clang,$(c))" ""
    ifeq "$(os)" "win"
        # Newer MS linker issues warnings if -Qsox is used:
        # "warning LNK4224: /COMMENT is no longer supported;  ignored"
        # so let us comment it out (and delete later).
        # ifneq "$(arch)" "32e"
        #     c-flags   += -Qsox
        #     cxx-flags += -Qsox
        # endif
        fort-flags += -Qsox
    else
        # For unknown reason, icc and ifort on mac does not accept this option.
        ifneq "$(filter lin lrb,$(os))" ""
            c-flags    += -sox
            cxx-flags  += -sox
            fort-flags += -sox
        endif
    endif
endif

ifeq "$(os)" "lrb"
    c-flags    += -mmic
    cxx-flags  += -mmic
    fort-flags += -mmic
    ld-flags   += -mmic
    as-flags   += -mmic
    cpp-flags  += -mmic
endif

# Exception handling.
ifeq "$(os)" "win"
    # ??? Enable exception handling?
    ifeq "$(LINK_TYPE)" "dyna"
        c-flags   += -EHsc
        cxx-flags += -EHsc
    endif
else
    # Disable exception handling.
    c-flags   += -fno-exceptions
    cxx-flags += -fno-exceptions
endif

# Disable use of EBP as general purpose register.
ifeq "$(os)" "win"
    ifeq "$(arch)" "32"
        c-flags   += -Oy-
        cxx-flags += -Oy-
    endif
endif

ifeq "$(os)" "lin"
    c-flags   += -Wsign-compare
    cxx-flags += -Wsign-compare
    ld-flags  += -Wsign-compare
    ifeq "$(filter gcc clang,$(c))" ""
        c-flags   += -Werror
        cxx-flags += -Werror
        ld-flags  += -Werror
    endif
endif
ifeq "$(os)" "win"
    c-flags   += -WX
    cxx-flags += -WX
    ld-flags  += -WX:NO
endif

ifeq "$(os)" "lrb"
    # With "-ftls-model=initial-exec" the compiler generates faster code for static TLS
    # accesses, it generates slower calls to glibc otherwise. We don't use this
    # feature on Linux because it prevents dynamic loading (use of dlopen) of the library.
    # Reliable dynamic loading is more important than slightly faster access to TLS.
    # On Intel(R) Xeon Phi(TM) coprocessor we haven't encountered dynamic loading problem yet, so use faster
    # access to static TLS.
    c-flags   += -ftls-model=initial-exec
    cxx-flags += -ftls-model=initial-exec
    # disable streaming stores in order to work on A0 Si
    c-flags   += -opt-streaming-stores never
    cxx-flags += -opt-streaming-stores never
endif

# Select C runtime.
ifeq "$(os)" "win"
    # Regardless of following -Zl option, we should specify -MT or -MTd, otherwise test-touch
    # wil fails due to unresolved reference "_errno".
    ifeq "$(OPTIMIZATION)" "on"
        c-flags   += -MT
        cxx-flags += -MT
    else
        c-flags   += -MTd
        cxx-flags += -MTd
    endif
    ifeq "$(LINK_TYPE)" "stat"
        # Do not emit C runtime library to object file. It will allows link OpenMP RTL with either static
        # or dynamic C runtime. Windows* OS specific, applicable only to static RTL.
        c-flags   += -Zl
        cxx-flags += -Zl
    endif
endif

ifeq "$(os)" "win"
    c-flags   += -W3
    cxx-flags += -W3
    # Disable warning: "... declared but never referenced"
    # Disable remark #5082: Directive ignored - Syntax error, found IDENTIFIER 'LRB'...
    fort-flags   += -Qdiag-disable:177,5082
    c-flags      += -Qdiag-disable:177
    cxx-flags    += -Qdiag-disable:177
endif

ifeq "$(CPLUSPLUS)" "on"
    ifeq "$(os)" "win"
        c-flags   += -TP
    else ifeq "$(arch)" "ppc64"
    # c++0x on ppc64 linux removes definition of preproc. macros, needed in .hs
      c-flags   += -x c++ -std=gnu++0x
    else
        ifneq "$(filter gcc clang,$(c))" ""
            c-flags   += -x c++ -std=c++0x
        else
            c-flags   += -Kc++
        endif
    endif
endif

# --- Linker options ---

ifeq "$(os)" "lin"
    ifneq "$(LIB_TYPE)" "stub"
        ifeq "$(ld)" "ld"
            # Warn about non-PIC code presence
            ld-flags += --warn-shared-textrel
            ld-flags += -fini=__kmp_internal_end_fini
            ld-flags += -lpthread
        else # $(c) or $(cxx)
            ld-flags += -Wl,--warn-shared-textrel
            ld-flags += -Wl,-fini=__kmp_internal_end_fini
            ld-flags += -pthread
        endif
    endif
    ifeq "$(ld)" "$(c)"
        ld-flags += -fPIC
        ifeq "$(DEBUG_INFO)" "on"
            ld-flags += -g
        endif
        ifeq "$(OPTIMIZATION)" "off"
            ld-flags += -O0
        endif
        ld-flags += -Wl,--version-script=$(src_dir)exports_so.txt
    else
        ld-flags += --version-script=$(src_dir)exports_so.txt
    endif
    ifeq "$(ld)" "$(c)"
        # to remove dependency on libimf, libsvml, libintlc:
        ifeq "$(c)" "icc"
            ld-flags-dll += -static-intel
        endif
        ld-flags-dll += -Wl,--as-needed
        # to remove dependency on libgcc_s:
        ifeq "$(c)" "gcc"
            ld-flags-dll += -static-libgcc
            ifneq "$(omp_os)" "freebsd"
                ld-flags-extra += -Wl,-ldl
            endif
        endif
        ifeq "$(c)" "clang"
            ifneq "$(omp_os)" "freebsd"
                ld-flags-extra += -Wl,-ldl
            endif
        endif
        ifeq "$(arch)" "32"
            ifeq "$(filter gcc clang,$(c))" ""
            # to workaround CQ215229 link libirc_pic manually
            ld-flags-extra += -lirc_pic
            endif
        endif
        ifeq "$(filter 32 32e 64 ppc64,$(arch))" ""
            ld-flags-extra += $(shell pkg-config --libs libffi)
        endif
    else
        ifeq "$(arch)" "32e"
            # ???
            ld-flags += -Bstatic -L/usr/lib64 -lc_nonshared -Bdynamic
        endif
    endif
endif

ifeq "$(os)" "lrb"
  ifeq "$(ld)" "ld"
    ifneq "$(LIB_TYPE)" "stub"
        ld-flags += -lthr
        ld-flags += -fini=__kmp_internal_end_atexit
        # Warn about non-PIC code presence
        ld-flags += --warn-shared-textrel
    endif
    ld-flags += --version-script=$(src_dir)exports_so.txt
  endif
  ifeq "$(ld)" "$(c)"
    ld-flags += -Wl,--warn-shared-textrel
    ld-flags += -Wl,--version-script=$(src_dir)exports_so.txt
    ld-flags += -static-intel
    # Don't link libcilk*.
    ld-flags += -no-intel-extensions
    # Discard unneeded dependencies.
    ld-flags += -Wl,--as-needed
#    ld-flags += -nodefaultlibs
    # To check which libraries the compiler links comment above line and uncomment below line
#    ld-flags += -\#
    # link libraries in the order the icc compiler uses (obtained using "icc -shared -#" command line)
    # Compiler 20101017 uses "-lintlc -lthr -lc -lintlc -lirc_s" sequence, we follow it:
#    ld-flags += -lintlc
    ifneq "$(LIB_TYPE)" "stub"
        ld-flags += -pthread
        ifeq "$(MIC_OS)" "lin"
            ld-flags += -ldl
        endif
    endif
  endif
endif

ifeq "$(os)" "mac"
    ifeq "$(c)" "icc"
        ld-flags += -no-intel-extensions
    endif
    ld-flags += -single_module
    ld-flags += -current_version $(VERSION).0 -compatibility_version $(VERSION).0
endif

ifeq "$(os)" "win"
    ld-flags += -incremental:no
    ld-flags += -version:$(VERSION).0
endif

# --------------------------------------------------------------------------------------------------
# Project-specific preprocessor definitions.
# --------------------------------------------------------------------------------------------------

cpp-flags += -D KMP_ARCH_STR="\"$(call legal_arch,$(arch))\""

ifeq "$(os)" "win"
    cpp-flags += -D _WINDOWS -D _WINNT -D _WIN32_WINNT=0x0501
    # 0x0501 means Windows* XP* OS or Windows* Server 2003* OS or later.
    # We need this for GetModuleHanleEx function.
    ifeq "$(LINK_TYPE)" "dyna"
        cpp-flags += -D _USRDLL
    endif
else # lin, lrb or mac
    cpp-flags += -D _GNU_SOURCE
    cpp-flags += -D _REENTRANT
endif

# TODO: DIAG leads to DEBUG. Confusing a bit. Raname KMP_DEBUG to KMP_DIAG?
ifeq "$(DIAG)" "on"
    cpp-flags += -D KMP_DEBUG
endif
ifeq "$(COVERAGE)" "on"
    cpp-flags += -D COVER
endif
# Assertions in OMP RTL code are controlled by two macros: KMP_DEBUG enables or disables assertions
# iff KMP_USE_ASSERT is defined. If KMP_USE_ASSERT is not defined, assertions disabled regardless of
# KMP_DEBUG. It was implemented for code coverage -- to have debug build with no assertion, but it
# does not have much effect. TODO: Remove macro.
ifeq "$(COVERAGE)" "off"
    cpp-flags += -D KMP_USE_ASSERT
endif

cpp-flags += -D BUILD_I8
ifneq "$(os)" "win"
    cpp-flags += -D BUILD_TV
endif
cpp-flags += -D KMP_LIBRARY_FILE=\"$(lib_file)\"
cpp-flags += -D KMP_VERSION_MAJOR=$(VERSION)

# customize ppc64 cache line size to 128, 64 otherwise
ifeq "$(arch)" "ppc64"
	cpp-flags += -D CACHE_LINE=128
else 
	cpp-flags += -D CACHE_LINE=64
endif

cpp-flags += -D KMP_ADJUST_BLOCKTIME=1
cpp-flags += -D BUILD_PARALLEL_ORDERED
cpp-flags += -D KMP_ASM_INTRINS
ifneq "$(os)" "lrb"
    cpp-flags += -D USE_LOAD_BALANCE
endif
ifneq "$(os)" "win"
    cpp-flags += -D USE_CBLKDATA
    # ??? Windows* OS: USE_CBLKDATA defined in kmp.h.
endif
ifeq "$(os)" "win"
    cpp-flags += -D KMP_WIN_CDECL
endif
ifeq "$(LINK_TYPE)" "dyna"
    cpp-flags += -D GUIDEDLL_EXPORTS
endif
ifeq "$(LIB_TYPE)" "stub"
    cpp-flags += -D KMP_STUB
endif
ifeq "$(VERSION)" "4"
else # 5
    ifeq "$(os)" "win"
    else
        cpp-flags += -D KMP_GOMP_COMPAT
    endif
endif

ifneq "$(filter 32 32e,$(arch))" ""
cpp-flags += -D KMP_USE_ADAPTIVE_LOCKS=1 -D KMP_DEBUG_ADAPTIVE_LOCKS=0
endif

# define compatibility with different OpenMP versions
have_omp_50=0
have_omp_41=0
have_omp_40=0
have_omp_30=0
ifeq "$(OMP_VERSION)" "50"
	have_omp_50=1
	have_omp_41=1
	have_omp_40=1
	have_omp_30=1
endif
ifeq "$(OMP_VERSION)" "41"
	have_omp_50=0
	have_omp_41=1
	have_omp_40=1
	have_omp_30=1
endif
ifeq "$(OMP_VERSION)" "40"
	have_omp_50=0
	have_omp_41=0
	have_omp_40=1
	have_omp_30=1
endif
ifeq "$(OMP_VERSION)" "30"
	have_omp_50=0
	have_omp_41=0
	have_omp_40=0
	have_omp_30=1
endif
cpp-flags += -D OMP_50_ENABLED=$(have_omp_50) -D OMP_41_ENABLED=$(have_omp_41)
cpp-flags += -D OMP_40_ENABLED=$(have_omp_40) -D OMP_30_ENABLED=$(have_omp_30)


# Using ittnotify is enabled by default.
USE_ITT_NOTIFY = 1
ifeq "$(os)-$(arch)" "win-64"
    USE_ITT_NOTIFY = 0
endif
ifeq "$(LINK_TYPE)" "stat"
    USE_ITT_NOTIFY = 0
endif
cpp-flags += -D USE_ITT_NOTIFY=$(USE_ITT_NOTIFY)
ifeq "$(USE_ITT_NOTIFY)" "0"
    # Disable all ittnotify calls.
    cpp-flags += -D INTEL_NO_ITTNOTIFY_API
else
    ifeq "$(os)" "win"
        ittnotify_static$(obj) : cpp-flags += -D UNICODE
    endif
endif
# Specify prefix to be used for external symbols. Prefix is required even if ITT Nofity turned off
# because we have some functions with __itt_ prefix (__itt_error_handler) and want prefix to be
# changed to __kmp_itt_.
cpp-flags += -D INTEL_ITTNOTIFY_PREFIX=__kmp_itt_


# Linux* OS: __declspec(thread) TLS is still buggy on static builds.
# Windows* OS: This define causes problems with LoadLibrary + declspec(thread) on Windows* OS. See CQ50564,
#     tests kmp_load_library_lib*.c, and the following MSDN reference:
#     http://support.microsoft.com/kb/118816
ifneq "$(filter lin lrb,$(os))" ""
    ifeq "$(LINK_TYPE)" "dyna"
        cpp-flags += -D KMP_TDATA_GTID
    else
        # AC: allow __thread in static build for Intel(R) 64, looks like it is
        # working there. It is broken on IA-32 architecture for RHEL4 and SLES9.
        ifeq "$(arch)" "32e"
            cpp-flags += -D KMP_TDATA_GTID
        endif
    endif
endif

# Intel compiler has a bug: in case of cross-build if used with
# -x assembler-with-cpp option, it defines macros for both architectures,
# host and tartget. For example, if compiler for IA-32 architecture
# runs on Intel(R) 64, it defines both __i386 and __x86_64. (Note it is a bug
# only if -x assembler-with-cpp is specified, in case of C files icc defines
# only one, target architecture). So we cannot autodetect target architecture
# within the file, and have to pass target architecture from command line.
ifneq "$(os)" "win"
    ifeq "$(arch)" "arm"
        z_Linux_asm$(obj) : \
		    cpp-flags += -D KMP_ARCH_ARM
    else ifeq "$(arch)" "ppc64" 
        z_Linux_asm$(obj) : \
			cpp-flags += -D KMP_ARCH_PPC64		    
    else
    	z_Linux_asm$(obj) : \
       		cpp-flags += -D KMP_ARCH_X86$(if $(filter 32e,$(arch)),_64)	
    endif
endif

# Defining KMP_BUILD_DATE for all files leads to warning "incompatible redefinition", because the
# same macro is also defined in omp.h. To avoid conflict, let us define macro with different name,
# _KMP_BUILD_TIME.
kmp_version$(obj) : cpp-flags += -D _KMP_BUILD_TIME="\"$(date)\""

# --- Macros for generate-def.pl ---

gd-flags += -D arch_$(arch)
gd-flags += -D $(LIB_TYPE)
ifeq "$(HAVE_QUAD)" "1"
    gd-flags += -D HAVE_QUAD
endif
ifeq "$(OMP_VERSION)" "40"
    gd-flags += -D OMP_40 -D OMP_30
else
    ifeq "$(OMP_VERSION)" "30"
        gd-flags += -D OMP_30
    endif
endif
ifneq "$(VERSION)" "4"
    gd-flags += -D msvc_compat
endif
ifeq "$(DIAG)" "on"
    gd-flags += -D KMP_DEBUG
endif

# --- Macro for expand-vars.pl ---

# $Revision and $Date often occur in file header, so define these variables to satisfy expand-vars.pl.
ev-flags += -D Revision="\$$Revision" -D Date="\$$Date"

# Various variables.
ev-flags += -D KMP_TYPE="$(call legal_type,$(LIB_TYPE))" -D KMP_ARCH="$(call legal_arch,$(arch))"
ev-flags += -D KMP_VERSION_MAJOR=$(VERSION) -D KMP_VERSION_MINOR=0 -D KMP_VERSION_BUILD=$(build)
ev-flags += -D KMP_BUILD_DATE="$(date)"
ev-flags += -D KMP_TARGET_COMPILER=$(TARGET_COMPILER)
ev-flags += -D KMP_DIAG=$(if $(filter on,$(DIAG)),1,0)
ev-flags += -D KMP_DEBUG_INFO=$(if $(filter on,$(DEBUG_INFO)),1,0)
ifeq "$(OMP_VERSION)" "40"
    ev-flags += -D OMP_VERSION=201307
else
    ifeq "$(OMP_VERSION)" "30"
        ev-flags += -D OMP_VERSION=201107
    else
        ev-flags += -D OMP_VERSION=200505
    endif
endif

# -- Options specified in command line ---

cpp-flags  += $(CPPFLAGS)
c-flags    += $(CFLAGS)
cxx-flags  += $(CXXFLAGS)
fort-flags += $(FFLAGS)
ld-flags   += $(LDFLAGS)

# --------------------------------------------------------------------------------------------------
# Files.
# --------------------------------------------------------------------------------------------------

# Library files. These files participate in all kinds of library.
lib_c_items :=      \
    kmp_ftn_cdecl   \
    kmp_ftn_extra   \
    kmp_version     \
    $(empty)
lib_cpp_items :=
lib_asm_items :=

# Files to be linked into import library.
imp_c_items :=

do_test_touch_mt := 1

ifeq "$(LIB_TYPE)" "stub"
    lib_c_items += kmp_stub
else # norm or prof
    lib_c_items +=                   \
        kmp_alloc                    \
        kmp_atomic                   \
        kmp_csupport                 \
        kmp_debug                    \
	kmp_itt                      \
        $(empty)
    ifeq "$(USE_ITT_NOTIFY)" "1"
        lib_c_items +=  ittnotify_static
    endif


    lib_cpp_items +=                 \
        kmp_environment              \
        kmp_error                    \
        kmp_global                   \
        kmp_i18n                     \
        kmp_io                       \
        kmp_runtime                  \
        kmp_settings                 \
        kmp_str                      \
        kmp_tasking                  \
        kmp_taskq                    \
        kmp_threadprivate            \
        kmp_utility                  \
        kmp_affinity                 \
        kmp_dispatch                 \
        kmp_lock                     \
        kmp_sched                    \
        $(empty)

ifeq "$(OMP_VERSION)" "40"
    lib_cpp_items += kmp_taskdeps
    lib_cpp_items += kmp_cancel
endif

    # OS-specific files.
    ifeq "$(os)" "win"
        lib_c_items += z_Windows_NT_util
        # Arch-specific files.
        lib_c_items   += z_Windows_NT-586_util
        lib_asm_items += z_Windows_NT-586_asm
        ifeq "$(LINK_TYPE)" "dyna"
            imp_c_items += kmp_import
            # for win_32/win_32e dynamic libguide40.dll,
            # build the shim lib instead
            ifeq "$(VERSION)" "4"
                ifneq "$(arch)" "64"
                    ifeq "$(LIB_TYPE)" "norm"
                    lib_c_items   = kmp_shim
                        lib_cpp_items =
                        lib_asm_items =
                        gd-flags += -D shim
                        # for some reason, test-touch-md is able to work with
                        # the build compiler's version of libiomp5md.dll, but
                        # test-touch-mt can't load it.
                        do_test_touch_mt := 0
                    endif
                endif
            endif
        endif
    else # lin, lrb or mac
        lib_c_items += z_Linux_util
        # GCC Compatibility files
        ifeq "$(VERSION)" "4"
        else # 5
            lib_c_items += kmp_gsupport
        endif
#        ifneq "$(arch)" "ppc64"
        lib_asm_items += z_Linux_asm
#	     endif
    endif
endif

lib_obj_files := $(sort $(addsuffix $(obj),$(lib_c_items) $(lib_cpp_items) $(lib_asm_items)))
imp_obj_files := $(sort $(addsuffix $(obj),$(imp_c_items) $(imp_cpp_items) $(imp_asm_items)))
dep_files     := $(sort $(addsuffix .d,$(lib_c_items) $(lib_cpp_items) $(imp_c_items) $(imp_cpp_items)))
i_files       := $(sort $(addsuffix .i,$(lib_c_items) $(lib_cpp_items) $(imp_c_items) $(imp_cpp_items)))


# --- Construct library file name ---

ifeq "$(VERSION)" "4"
    ifeq "$(LIB_TYPE)" "stub"
        _lib_item = libompstub
    else # norm or prof
        _lib_item = libguide
    endif
    ifeq "$(os)-$(LINK_TYPE)" "win-dyna"
        _lib_item += 40
    endif
    ifeq "$(LIB_TYPE)" "prof"
        _lib_item += _stats
    endif
else
    _lib_item = libiomp
    ifeq "$(LIB_TYPE)" "prof"
        _lib_item += prof
    endif
    ifeq "$(LIB_TYPE)" "stub"
        _lib_item += stubs
    endif
    _lib_item += $(VERSION)
    ifeq "$(os)" "win"
        ifeq "$(LINK_TYPE)" "dyna"
            _lib_item += md
        else
            _lib_item += mt
        endif
    endif
endif
# _lib_item is a list of space separated name parts. Remove spaces to form final name.
lib_item = $(subst $(space),,$(_lib_item))
ifeq "$(LINK_TYPE)" "dyna"
    lib_ext = $(dll)
else
    lib_ext = $(lib)
endif
lib_file  = $(lib_item)$(lib_ext)
ifeq "$(os)-$(LINK_TYPE)" "win-dyna"
    imp_file  = $(lib_item)$(lib)
    def_file  = $(lib_item).def
    res_file  = $(lib_item).res
    rc_file   = $(lib_item).rc
    # PDB file should be generated if: ( DEBIG_INFO is on ) OR ( we are building 32-bit normal
    # library AND version is 5 ).
    ifneq "$(filter on,$(DEBUG_INFO))$(filter norm-5,$(LIB_TYPE)-$(VERSION))" ""
        pdb_file = $(lib_item).pdb
    endif
endif
ifneq "$(filter lin lrb,$(os))" ""
    ifeq "$(LINK_TYPE)" "dyna"
      ifneq "$(DEBUG_INFO)" "on"
        dbg_file = $(lib_item).dbg
      endif
    else
        dbg_strip = "on"
    endif
endif

# --- Output files ---

out_lib_files  = $(addprefix $(out_lib_dir),$(lib_file) $(imp_file) $(pdb_file) $(dbg_file))
out_inc_files  = $(addprefix $(out_ptf_dir)include_compat/,iomp_lib.h)
out_mod_files  = \
    $(addprefix $(out_ptf_dir)include/,omp_lib.mod omp_lib_kinds.mod)
out_cmn_files  = \
    $(addprefix $(out_cmn_dir)include/,omp.h omp_lib.h omp_lib.f omp_lib.f90) \
    $(addprefix $(out_cmn_dir)include_compat/,iomp.h)
ifneq "$(out_lib_fat_dir)" ""
    out_lib_fat_files  = $(addprefix $(out_lib_fat_dir),$(lib_file) $(imp_file))
endif

# --- Special dependencies ---

# We have to encode dependencies on omp.h manually, because automatic dependency generation
# by compiler produces depedency on omp.h file located in compiler include directory.
kmp_csupport$(obj) : omp.h
kmp_stub$(obj)     : omp.h

# --------------------------------------------------------------------------------------------------
# External libraries to link in.
# --------------------------------------------------------------------------------------------------

# We (actually, our customers) do no want OpenMP RTL depends on external libraries, so we have to
# pick up some object files from libirc library (Intel compiler generate code with calls to libirc)
# and link them into OMP RTL.
# libipgo is required only for collecting code coverage data, but is is convenient to link in into
# OMP RTL as well, not to depend on extra libs and paths.

# libirc does matter only if Intel compiler is used.
ifneq "$(filter icc icl icl.exe,$(c))" ""

    ifneq "$(ICC_LIB_DIR)" ""
        icc_lib_dir := $(ICC_LIB_DIR)
    else
        #
        # Let us find path to Intel libraries first. (don't use tabs in these lines!)
        #
        icc_path := $(shell which $(c))
        $(call debug,icc_path)
        ifeq "$(words $(icc_path))" "0"
            $(error Path to "$(c)" not found, reported path: $(icc_path))
        endif
        ifneq "$(words $(icc_path))" "1"
            $(error Path to "$(c)" contains spaces: "$(icc_path)")
        endif
        ifeq "$(os)" "win"  # Windows* OS specific.
            # `which' can return path with backslashes. Convert them.
            icc_path := $(subst \,/,$(icc_path))
            # icc's "bin/" directory may be named as "Bin/" or even "BIN/". Convert it to lower case.
            icc_path := $(subst B,b,$(icc_path))
            icc_path := $(subst I,i,$(icc_path))
            icc_path := $(subst N,n,$(icc_path))
            $(call debug,icc_path)
        endif
        # icc 10.x directory layout:
        #         bin/
        #         lib/
        # icc 11.x directory layout:
        #         bin/{ia32,intel64}/
        #         lib/{ia32,intel64}/
        # icc 12.x directory layout:
        #         bin/{ia32,intel64}/
        #         compiler/lib/{ia32,intel64}/
        # Note: On OS X* fat libraries reside in lib/ directory. On other systems libraries are in
        # lib/<arch>/.
        icc_path_up1 := $(dir $(icc_path))
        icc_path_up2 := $(dir $(patsubst %/,%,$(icc_path_up1)))
        $(call debug,icc_path_up1)
        $(call debug,icc_path_up2)
        ifneq "$(filter %/bin/,$(icc_path_up1))" ""
            # Look like 10.x compiler.
            icc_lib_dir := $(patsubst %/bin/,%/lib/,$(icc_path_up1))
        else
            ifneq "$(filter %/bin/,$(icc_path_up2))" ""
                # It looks like 11.x or later compiler.
                ifeq "$(os)" "mac"
                    icc_lib12 := $(patsubst %/bin/,%/compiler/lib/,$(icc_path_up2))
                    ifneq "$(wildcard $(icc_lib12)libirc*$(lib))" ""
                        # 12.x
                        icc_lib_dir := $(icc_lib12)
                    else
                        # 11.x
                        icc_lib_dir := $(patsubst %/bin/,%/lib/,$(icc_path_up2))
                    endif
                else
                    icc_lib12 := $(patsubst %/bin/,%/compiler/lib/,$(icc_path_up2))$(notdir $(patsubst %/,%,$(icc_path_up1)))/
                    ifneq "$(wildcard $(icc_lib12)libirc*$(lib))" ""
                        # 12.x
                        icc_lib_dir := $(icc_lib12)
                    else
                        # 11.x
                        icc_lib_dir := $(patsubst %/bin/,%/lib/,$(icc_path_up2))$(notdir $(patsubst %/,%,$(icc_path_up1)))/
                    endif
                endif
            endif
        endif
        $(call debug,icc_lib_dir)
        ifeq "$(icc_lib_dir)" ""
            $(error Path to $(c) lib/ dir not found)
        endif
    endif

    #
    # Then select proper libraries.
    #
    ifeq "$(os)" "win"
        libirc  = $(icc_lib_dir)\libircmt$(lib)
        libipgo = $(icc_lib_dir)\libipgo$(lib)
    else # lin, lrb or mac
        ifeq "$(LINK_TYPE)" "dyna"
            # In case of dynamic linking, prefer libi*_pic.a libraries, they contains
            # position-independent code.
            libirc  = $(icc_lib_dir)libirc_pic$(lib)
            libipgo = $(icc_lib_dir)libipgo_pic$(lib)
            # If libi*_pic.a is not found (it is missed in older compilers), use libi*.a.
            ifeq "$(wildcard $(libirc))" ""
                libirc = $(icc_lib_dir)libirc$(lib)
            endif
            ifeq "$(wildcard $(libipgo))" ""
                libipgo = $(icc_lib_dir)libipgo$(lib)
            endif
        else
            libirc  = $(icc_lib_dir)libirc$(lib)
            libipgo = $(icc_lib_dir)libipgo$(lib)
        endif
    endif

    # Ok, now let us decide when linked
    # Linux* OS:
    # We link in libraries to static library only.
    ifeq "$(os)-$(LINK_TYPE)" "lin-stat"
        linked_in_libs += libirc
    endif
    # OS X*:
    # The trick is not required in case of dynamic library, but on Intel(R) 64 architecture we have a
    # problem: libirc.a is a fat, so linker (libtool) produces fat libguide.dylib... :-( (Only
    # functions from libirc are presented for both architectures, libguide functions are for Intel(R) 64
    # only.) To avoid this undesired effect, libirc trick is enabled for both static and dynamic
    # builds. Probably we can instruct libtool to produce "thin" (not fat) library by using
    # -arch_only option...
    ifeq "$(os)" "mac"
        linked_in_libs += libirc
    endif
    # Windows* OS:
    # The trick is required only in case of static OMP RTL. In case of dynamic OMP RTL linker does
    # the job.
    ifeq "$(os)-$(LINK_TYPE)" "win-stat"
        linked_in_libs += libirc
    endif

    ifeq "$(COVERAGE)" "on"
        linked_in_libs += libipgo
    endif

endif

# --------------------------------------------------------------------------------------------------
# Main targets.
# --------------------------------------------------------------------------------------------------

all    : lib inc mod
lib    : tests $(out_lib_files)
inc    : $(out_inc_files)
mod    : $(out_mod_files)
clean  :
	$(rm) $(out_lib_files) $(out_lib_fat_files)
	$(rm) $(out_inc_files) $(out_mod_files)

# --------------------------------------------------------------------------------------------------
# Building library.
# --------------------------------------------------------------------------------------------------

$(lib_file) : $(if $(dbg_file),stripped,unstripped)/$(lib_file)
	$(target)
	$(cp) $< $@

ifneq "$(dbg_file)" ""
    $(dbg_file) : unstripped/$(dbg_file)
	$(target)
	$(cp) $< $@
endif

ifneq "$(filter lin lrb,$(os))" ""
    lib_file_deps = $(if $(linked_in_libs),required/.objs,$(lib_obj_files))
endif
ifeq "$(os)" "mac"
    lib_file_deps = iomp$(obj)
endif
ifeq "$(os)" "win"
    lib_file_deps = $(if $(linked_in_libs),wiped/.objs,$(lib_obj_files))
endif

# obj_dep_files -- object files, explicitly specified in dependency list. Other files (non-object)
# are filtered out.
obj_deps_files = $(filter %$(obj),$^)
# obj_deps_flags -- object files corresponding to flags, specified in dependency list. Flag is a
# special file like "required/.objs". Flag file is replaced with a list of all object files in flag
# directory, for example, "required/*.o"
obj_deps_flags = $(addsuffix *$(obj),$(dir $(filter %/.objs,$^)))
# obj_deps_all -- list of all object files specified in dependency list, either explicit or found
# in flagged directories.
obj_deps_all   = $(obj_deps_files) $(obj_deps_flags)

unstripped/$(lib_file).lst : $(lib_file_deps) unstripped/.dir .rebuild
	$(target)
	echo $(obj_deps_all) > $@

ifeq "$(os)-$(LINK_TYPE)" "lin-dyna"
    $(lib_file) : exports_so.txt
endif

# Copy object files, wiping out references to libirc library. Object files (ours and extracted
# from libirc.lib) have "-defaultlib:libirc.lib" linker directive, so linker will require libirc.lib
# regardless of absence of real dependency. Actually, this rule is required only on Windows* OS, but
# there is no Windows* OS-specific commands, so I omit conditions to keep code shorter and be able test
# the rule on Linux* OS.
# Note: If we are not going to pick up objects from libirc, there is no point in wiping out
# libirc references.
# Addition: Wipe also references to C++ runtime (libcpmt.lib) for the same reason: sometimes C++
# runtime routines are not actually used, but compiler puts "-defaultlib:libcpmt.lib" directive to
# object file. Wipe it out, if we have real dependency on C++ runtime, test-touch will fail.
wiped/.objs : required/.objs \
    $(tools_dir)wipe-string.pl wiped/.dir .rebuild
	$(target)
	$(rm) $(dir $@)*$(obj)
    ifeq "$(os)" "win"
	$(perl) $(tools_dir)wipe-string.pl --quiet \
	    --wipe-regexp="(-|/)(defaultlib|DEFAULTLIB):\"(libir|libc|LIBC|OLDN|libmm|libde|svml).*?\"" \
	    --target-directory=$(dir $@) $(obj_deps_all)
    else
	$(perl) $(tools_dir)wipe-string.pl --quiet \
	    --wipe-regexp="(-|/)(defaultlib|DEFAULTLIB):\"(libir|libc|LIBC|OLDN).*?\"" \
	    --target-directory=$(dir $@) $(obj_deps_all)
    endif
	$(touch) $@

# required-objects.pl uses "objcopy" utility to rename symbols in object files. On Linux* OS this is a
# standard utility (from binutils package). On Windows* OS we provides homebrew implementation (very
# limited, but enough for our purposes).
ifeq "$(os)" "win"
    objcopy = objcopy$(exe)
endif

# required/ is a directory containing OMP RTL object files and really required files from external
# libraries. required/.obj is a flag. If this file present, it means all required objects already
# in place. Note, required-objects.pl copies files to specified directory. It is necessary, because
# object files are edited during copying -- symbols defined in external object files are renamed.
required/.objs : $(lib_obj_files) $(addsuffix /.objs,$(linked_in_libs)) \
    $(tools_dir)required-objects.pl $(objcopy) required/.dir .rebuild
	$(target)
	$(rm) $(dir $@)*$(obj)
	$(perl) $(tools_dir)required-objects.pl --quiet $(oa-opts) --prefix=__kmp_external_ \
	    --base $(obj_deps_files) --extra $(obj_deps_flags) --copy-all=$(dir $@)
	$(touch) $@

# Extracting object files from libirc. File "libirc/.obj" is a flag. If the file present, make
# thinks files are extracted.
ifneq "$(libirc)" ""
    libirc/.objs : $(libirc) \
	$(tools_dir)extract-objects.pl libirc/.dir .rebuild
	    $(target)
	    $(rm) $(dir $@)*$(obj)
	    $(perl) $(tools_dir)extract-objects.pl --quiet $(oa-opts) --output=$(dir $@) $<
	    $(touch) $@
endif

# Extracting object files from libipgo. File "/libipgo/.obj" is a flag. If the file present, make
# thinks objects are extracted.
ifneq "$(libipgo)" ""
    libipgo/.objs : $(libipgo) \
	$(tools_dir)extract-objects.pl libipgo/.dir .rebuild
	    $(target)
	    $(rm) $(dir $@)*$(obj)
	    $(perl) $(tools_dir)extract-objects.pl --quiet $(oa-opts) --output=$(dir $@) $<
	    $(touch) $@
endif


stripped/$(lib_file) : unstripped/$(lib_file) $(dbg_file) stripped/.dir .rebuild
	$(target)
	objcopy --strip-debug $< $@.tmp
	objcopy --add-gnu-debuglink=$(dbg_file) $@.tmp $@

ifeq "$(os)" "mac"

    # These targets are under condition because of some OS X*-specific ld and nm options. For
    # example, GNU nm does not accept -j, GNU ld does not know -filelist.

    # iomp.o is a big object file including all the OMP RTL object files and object files from
    # external libraries (like libirc). It is partially linked, references to external symbols
    # (e. g. references to libirc) already resolved, symbols defined in external libraries are
    # hidden by using -unexported-symbol-list and -non_global_symbols_strip_list linker options
    # (both options are required).
    # AC: 2012-04-12: after MAC machines upgrade compiler fails to create object, so use linker instead
ifeq "$(mac_os_new)" "1"
    iomp$(obj) : $(lib_obj_files) external-symbols.lst external-objects.lst .rebuild
	    $(target)
	    ld -r -unexported_symbols_list external-symbols.lst \
		-non_global_symbols_strip_list external-symbols.lst \
		-filelist external-objects.lst \
		-o $@ $(obj_deps_files)
else
    iomp$(obj) : $(lib_obj_files) external-symbols.lst external-objects.lst .rebuild
	    $(target)
	    $(c) -r -nostartfiles -static-intel  -no-intel-extensions \
		-Wl,-unexported_symbols_list,external-symbols.lst \
		-Wl,-non_global_symbols_strip_list,external-symbols.lst \
		-filelist external-objects.lst \
		-o $@ $(obj_deps_files)
endif

    # external-objects.lst is a list of object files extracted from external libraries, which should
    # be linked into iomp.o. kmp_dummy.o is added to the list to avoid empty list -- OS X* utilities
    # nm and ld do not like empty lists.
    external-objects.lst : $(lib_obj_files) $(addsuffix /.objs,$(linked_in_libs)) kmp_dummy$(obj) \
	$(tools_dir)required-objects.pl .rebuild
	    $(target)
	    $(perl) $(tools_dir)required-objects.pl $(oa-opts) \
		--base $(obj_deps_files) --extra $(obj_deps_flags) --print-extra > $@.tmp
	    echo "kmp_dummy$(obj)" >> $@.tmp
	    mv $@.tmp $@

    # Prepare list of symbols in external object files. We will hide all these symbols.
    # Note: -j is non-GNU option which means "Just display the symbol names (no value or type)."
    external-symbols.lst : external-objects.lst .rebuild
	    $(target)
	    nm -goj $$(cat external-objects.lst) > $@.0.tmp
	    cut -f2 -d" " $@.0.tmp > $@.1.tmp
	    mv $@.1.tmp $@

endif # mac

# Import library tricks are Windows* OS-specific.
ifeq "$(os)" "win"

    import$(lib) : $(lib_item)$(dll)

    # Change the name of import library produced by the linker, we will combine it with some more
    # object files to produce "extended import lib".
    $(lib_item)$(dll) : imp_file := import$(lib)

    # Default rule "c to obj" will compile sources with -MT option, which is not desired.
    # Cancel effect of -MT with -Zl.
    # AC: Currently we only have one object that does not need any special
    #     compiler options, so use minimal set. With standard set of options we used
    #     there were problems with debug info leaked into the import library
    #     with this object (bug report #334565).
    $(imp_obj_files) : c-flags := -Zl -nologo -c

    $(imp_file).lst : $(imp_obj_files) import$(lib) .rebuild
	    $(target)
	    echo $(filter-out .rebuild,$^) > $@

endif

kmp_i18n_id.inc : en_US.txt \
    $(tools_dir)message-converter.pl .rebuild
	$(target)
	$(perl) $(tools_dir)message-converter.pl $(oa-opts) --prefix=kmp_i18n --enum=$@ $<

kmp_i18n_default.inc : en_US.txt \
    $(tools_dir)message-converter.pl .rebuild
	$(target)
	$(perl) $(tools_dir)message-converter.pl $(oa-opts) --prefix=kmp_i18n --default=$@ $<

# Rebuilt kmp_version.o on any change to have actual build time string always updated.
kmp_version$(obj): $(filter-out kmp_version$(obj),$(lib_obj_files) $(imp_obj_files))

$(def_file) : dllexports \
    $(tools_dir)generate-def.pl .rebuild
	$(target)
	$(perl) $(tools_dir)generate-def.pl $(gd-flags) -o $@ $<

libiomp.rc : libiomp.rc.var kmp_version.c
libiomp.rc : ev-flags += -D KMP_FILE=$(lib_file)

$(rc_file) : libiomp.rc .rebuild
	$(target)
	$(cp) $< $@

kmp_dummy.c : .rebuild
	$(target)
	echo "void __kmp_dummy() {}" > $@

# --------------------------------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------------------------------

# --- test-touch ---

# test-touch is not available for lrb.
ifneq "$(os)" "lrb"

    # Compile a simple C test and link it with the library. Do it two times: the first link gives us
    # clear message if there are any problems, the second link run in verbose mode, linker output
    # searched for "libirc" string -- should not be any libirc references. Finally, test executable
    # run with KMP_VERBOSE=1.

    ifeq "$(os)" "win"
        ifneq "$(do_test_touch_mt)" "0"
            test_touch_items += test-touch-md test-touch-mt
        else
            test_touch_items += test-touch-md
        endif
    else
        test_touch_items += test-touch-rt
    endif

    force-test-touch : $(addsuffix /.force,$(test_touch_items)) $(addsuffix /.test,$(test_touch_items))
    test-touch       : $(addsuffix /.test,$(test_touch_items))

    tt-exe-file = $(dir $@)test-touch$(exe)
    ifeq "$(os)" "win"
        # On Windows* OS the test quality is problematic, because LIB environment variable is set up for
        # Intel compiler so Microsoft compiler is able to find libirc if it is specified in defaultlib
        # directive within object file... This disadvantage is compensated by grepping verbose link
        # output for "libirc" string.
        tt-c            = cl
        tt-c-flags     += -nologo
        ifeq "$(OPTIMIZATION)" "on"
            tt-c-flags-mt = -MT
            tt-c-flags-md = -MD
        else
            tt-c-flags-mt = -MTd
            tt-c-flags-md = -MDd
        endif
        ifeq "$(LINK_TYPE)" "stat"
            tt-libs  += $(lib_file)
        else
            tt-libs  += $(imp_file)
        endif
        ifneq "$(arch)" "32"
            # To successfully build with VS2008
            # tt-libs += bufferoverflowu.lib
            # Preventing "unresolved external symbol __security_cookie" (and
            # "... __security_check_cookie") linker errors on win_32e and win_64.
        endif
        tt-c-flags  += -Fo$(dir $@)test-touch$(obj) -Fe$(tt-exe-file)
        tt-ld-flags += -link
        # Some Posix but non-ISO functions (like strdup) are defined in oldnames.lib, which is used
        # implicitly. Drop oldnames.lib library, so we can catch
        tt-ld-flags += -nodefaultlib:oldnames
        ifeq "$(arch)" "32"
            tt-ld-flags += -safeseh
        endif
        tt-ld-flags-v += -verbose
    else # lin or mac
        # On Linux* OS and OS X* the test is good enough because GNU compiler knows nothing
        # about libirc and Intel compiler private lib directories, but we will grep verbose linker
        # output just in case.
        tt-c        = cc
        ifeq "$(os)" "lin"    # GCC on OS X* does not recognize -pthread.
            tt-c-flags  += -pthread
        endif
        tt-c-flags += -o $(tt-exe-file)
        ifneq "$(filter 32 32e 64,$(arch))" ""
            tt-c-flags += $(if $(filter 64,$(arch)),,$(if $(filter 32,$(arch)),-m32,-m64))
        endif
        tt-libs    += $(lib_file)
        ifeq "$(os)-$(COVERAGE)-$(LINK_TYPE)" "lin-on-stat"
            # Static coverage build on Linux* OS fails due to unresolved symbols dlopen, dlsym, dlclose.
            # Explicitly add dl library to avoid failure.
            tt-ld-flags += -ldl
        endif
        ifeq "$(os)" "mac"
            tt-ld-flags-v += -Wl,-t
            tt-env        += DYLD_LIBRARY_PATH=".:$(DYLD_LIBRARY_PATH)"
        else # lin
            tt-ld-flags-v += -Wl,--verbose
            tt-env        += LD_LIBRARY_PATH=".:$(LD_LIBRARY_PATH)"
        endif
    endif
    tt-c-flags += $(tt-c-flags-rt)
    tt-env     += KMP_VERSION=1
    tt-i        = $(if $(filter off,$(TEST_TOUCH)),-)

    ifndef test-touch-commands
        # The first building gives short and clear error message in case of any problem.
        # The second building runs linker in verbose mode and saves linker output for grepping.
      define test-touch-commands
	    $(rm) $(dir $@)*
	    $(tt-i)$(tt-c) $(tt-c-flags) $< $(tt-libs) $(tt-ld-flags)
	    $(rm) $(tt-exe-file)
	    $(tt-i)$(tt-c) $(tt-c-flags) \
		$< $(tt-libs) \
		$(tt-ld-flags) $(tt-ld-flags-v) \
		> $(dir $@)build.log 2>&1
	    $(tt-i)$(tt-env) $(tt-exe-file)
	    $(tt-i)grep -i -e "[^_]libirc" $(dir $@)build.log > $(dir $@)libirc.log; \
		[ $$? -eq 1 ]
      endef
    endif

    test-touch-rt/.test : tt-c-flags-rt =
    test-touch-mt/.test : tt-c-flags-rt = $(tt-c-flags-mt)
    test-touch-md/.test : tt-c-flags-rt = $(tt-c-flags-md)

    test-touch-rt/.test : test-touch.c $(lib_file) test-touch-rt/.dir .rebuild
	    $(target)
	    $(test-touch-commands)
	    $(touch) $@
    test-touch-mt/.test : test-touch.c $(lib_file) $(imp_file) test-touch-mt/.dir .rebuild
	    $(target)
	    $(test-touch-commands)
	    $(touch) $@
    test-touch-md/.test : test-touch.c $(lib_file) $(imp_file) test-touch-md/.dir .rebuild
	    $(target)
	    $(test-touch-commands)
	    $(touch) $@

endif

# --- test-relo ---

# But test-relo does actual work only on Linux* OS and
# Intel(R) Many Integrated Core Architecture in case of dynamic linking.
ifeq "$(if $(filter lin lrb,$(os)),os)-$(LINK_TYPE)" "os-dyna"

    # Make sure dynamic library does not contain position-dependent code.
    force-test-relo : test-relo/.force test-relo/.test
    test-relo       : test-relo/.test

    test-relo/.test : $(lib_item)$(dll) test-relo/.dir .rebuild
	    $(target)
	    readelf -d $< > $(dir $@)readelf.log
	    grep -e TEXTREL $(dir $@)readelf.log; [ $$? -eq 1 ]
	    $(touch) $@

endif

# --- test-execstack ---

# But test-execstack does actual work only on Linux* OS in case of dynamic linking.
# TODO: Enable it on Intel(R) Many Integrated Core Architecture as well.
ifeq "$(if $(filter lin,$(os)),os)-$(LINK_TYPE)" "os-dyna"

    tests += test-execstack

    # Make sure stack is not executable.
    force-test-execstack : test-execstack/.force test-execstack/.test
    test-execstack       : test-execstack/.test

    test-execstack/.test : $(lib_item)$(dll) test-execstack/.dir .rebuild
	    $(target)
	    $(perl) $(tools_dir)check-execstack.pl $<
	    $(touch) $@

endif

# --- test-instr ---

# But test-instr does actual work only on Intel(R) Many Integrated Core Architecture.
ifeq "$(os)" "lrb"

    # Make sure dynamic library does not contain position-dependent code.
    force-test-instr : test-instr/.force test-instr/.test
    test-instr       : test-instr/.test

    test-instr/.test : $(lib_file) $(tools_dir)check-instruction-set.pl test-instr/.dir .rebuild
	    $(target)
	    $(perl) $(tools_dir)check-instruction-set.pl $(oa-opts) --show --mic-arch=$(MIC_ARCH) --mic-os=$(MIC_OS) $<
	    $(touch) $@

endif

# --- test-deps ---

# test-deps does actual work for dymanic linking (all OSes), and Windows* OS (all linking types).
ifneq "$(filter %-dyna win-%,$(os)-$(LINK_TYPE))" ""

    force-test-deps : test-deps/.force test-deps/.test
    test-deps       : test-deps/.test

    td_exp =
    ifeq "$(os)" "lin"
        ifeq "$(arch)" "32"
            td_exp += libc.so.6
            td_exp += ld-linux.so.2
        endif
        ifeq "$(arch)" "32e"
            td_exp += libc.so.6
            td_exp += ld-linux-x86-64.so.2
        endif
        ifeq "$(arch)" "64"
            td_exp += libc.so.6.1
        endif
        ifeq "$(arch)" "arm"
            td_exp += libc.so.6
            td_exp += ld-linux-armhf.so.3
        endif
        ifeq "$(arch)" "ppc64"
            td_exp += libc.so.6
            td_exp += ld64.so.1
        endif
        td_exp += libdl.so.2
        td_exp += libgcc_s.so.1
        ifeq "$(filter 32 32e 64 ppc64,$(arch))" ""
            td_exp += libffi.so.6
            td_exp += libffi.so.5
        endif
        ifneq "$(LIB_TYPE)" "stub"
            td_exp += libpthread.so.0
        endif
    endif
    ifeq "$(os)" "lrb"
        ifeq "$(MIC_OS)" "lin"
            ifeq "$(MIC_ARCH)" "knf"
                td_exp += "ld-linux-l1om.so.2"
                td_exp += libc.so.6
                td_exp += libpthread.so.0
                td_exp += libdl.so.2
                td_exp += libgcc_s.so.1
            endif
            ifeq "$(MIC_ARCH)" "knc"
                td_exp += "ld-linux-k1om.so.2"
                td_exp += libc.so.6
                td_exp += libdl.so.2
                td_exp += libpthread.so.0
            endif
        endif
        ifeq "$(MIC_OS)" "bsd"
            td_exp += libc.so.7
            td_exp += libthr.so.3
            td_exp += libunwind.so.5
        endif
    endif
    ifeq "$(os)" "mac"
#        td_exp += /usr/lib/libgcc_s.1.dylib
        td_exp += /usr/lib/libSystem.B.dylib
    endif
    ifeq "$(os)" "win"
        ifeq "$(LINK_TYPE)" "dyna"
            td_exp += kernel32.dll
        else
            td_exp += uuid
        endif
    endif
    ifeq "$(omp_os)" "freebsd"
        td_exp =
        td_exp += libc.so.7
        td_exp += libthr.so.3
        td_exp += libunwind.so.5
    endif

    test-deps/.test : $(lib_file) $(tools_dir)check-depends.pl test-deps/.dir .rebuild
	    $(target)
	    $(td-i)$(perl) $(tools_dir)check-depends.pl $(oa-opts) \
		$(if $(td_exp),--expected="$(subst $(space),$(comma),$(td_exp))") $<
	    $(touch) $@

endif


# --------------------------------------------------------------------------------------------------
# Fortran files.
# --------------------------------------------------------------------------------------------------
ifeq "$(TARGET_COMPILER)" "11"
    omp_lib_f = omp_lib.f
endif
ifeq "$(TARGET_COMPILER)" "12"
    omp_lib_f = omp_lib.f90
endif
ifeq "$(omp_lib_f)" ""
    $(error omp_lib_f is not defined)
endif
omp_lib.mod omp_lib_kinds.mod : $(omp_lib_f) .rebuild
	$(target)
	$(fort) $(fort-flags) $<

omp_lib.h  : ev-flags += -D KMP_INT_PTR_KIND="int_ptr_kind()"
iomp_lib.h : ev-flags += -D KMP_INT_PTR_KIND=$(if $(filter 32,$(arch)),4,8)

# --------------------------------------------------------------------------------------------------
# Common files.
# --------------------------------------------------------------------------------------------------

common : $(out_cmn_files)

clean-common :
	$(rm) $(out_cmn_files)

# --------------------------------------------------------------------------------------------------
# Dependency files and common rules.
# --------------------------------------------------------------------------------------------------

.PHONY : dep
dep    : $(dep_files)
	$(target)

include $(LIBOMP_WORK)src/rules.mk

# Initiate rebuild if any of makefiles or build sript is changed.
# When developing makefiles, it is useful to comment it, otherwise make will perform full rebuild
# on every change of makefiles.
.rebuild : $(MAKEFILE_LIST) $(tools_dir)build.pl $(tools_dir)lib/Build.pm

ifeq "$(clean)" ""
    # Do not include dependency files if "clean" goal is specified.
    -include $(dep_files)
endif

# end of file #
