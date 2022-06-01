# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Defines variables that use selects to configure LLVM based on platform."""

def maybe_one(b):
    """Return "1" if b is True, else empty string"""
    return "1" if b else ""

def maybe_value(b, val):
    """Return val if b is True, else empty string"""
    return val if b else ""

def get_config_values(host_os, arch, triple):
    """Return CMake variables for config.h

    The variables for config.h are mostly LLVM-internal portability defines.
    """
    is_win = host_os == "win"
    is_posix = not is_win
    is_mac = host_os == "mac"
    is_linux = host_os == "linux"

    one_if_linux = maybe_one(is_linux)
    one_if_posix = maybe_one(is_posix)
    one_if_mac = maybe_one(is_mac)
    one_if_win = maybe_one(is_win)

    if is_win:
        shlib_ext = ".dll"
    elif is_mac:
        shlib_ext = ".dylib"
    else:
        shlib_ext = ".so"

    # Common variables.
    return [
        "BUG_REPORT_URL=https://github.com/llvm/llvm-project/issues/",
        "ENABLE_BACKTRACES=1",
        "ENABLE_CRASH_OVERRIDES=1",
        "HAVE_CRASHREPORTERCLIENT_H=",
        "HAVE_DECL_FE_ALL_EXCEPT=1",
        "HAVE_DECL_FE_INEXACT=1",
        "LLVM_ENABLE_CRASH_DUMPS=",
        "HAVE_ERRNO_H=1",
        "HAVE_FCNTL_H=1",
        "HAVE_FENV_H=1",
        "HAVE_FFI_CALL=",
        "HAVE_FFI_FFI_H=",
        "HAVE_FFI_H=",
        "HAVE_LIBPFM=",
        "HAVE_LIBPSAPI=",
        "HAVE_MALLCTL=",
        "HAVE_SIGNAL_H=1",
        "HAVE_STD_IS_TRIVIALLY_COPYABLE=1",
        "HAVE_STRERROR=1",
        "HAVE_SYS_STAT_H=1",
        "HAVE_SYS_TYPES_H=1",
        "HAVE_VALGRIND_VALGRIND_H=",
        "HAVE__ALLOCA=",
        "HAVE___ALLOCA=",
        "HAVE___ASHLDI3=",
        "HAVE___ASHRDI3=",
        "HAVE___CHKSTK=",
        "HAVE___CHKSTK_MS=",
        "HAVE___CMPDI2=",
        "HAVE___DIVDI3=",
        "HAVE___FIXDFDI=",
        "HAVE___FIXSFDI=",
        "HAVE___FLOATDIDF=",
        "HAVE___LSHRDI3=",
        "HAVE___MAIN=",
        "HAVE___MODDI3=",
        "HAVE___UDIVDI3=",
        "HAVE___UMODDI3=",
        "HAVE____CHKSTK=",
        "HAVE____CHKSTK_MS=",
        "HOST_LINK_VERSION=",
        "LIBPFM_HAS_FIELD_CYCLES=",
        "LLVM_TARGET_TRIPLE_ENV=",
        "LLVM_VERSION_INFO=",
        "LLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO=1",
        "LLVM_WINDOWS_PREFER_FORWARD_SLASH=",
        "PACKAGE_BUGREPORT=https://github.com/llvm/llvm-project/issues/",
        "PACKAGE_NAME=LLVM",
        "PACKAGE_STRING=LLVM git",
        "PACKAGE_VERSION=git",
        "PACKAGE_VENDOR=",
        "RETSIGTYPE=void",
        "LLVM_GISEL_COV_ENABLED=",
        "LLVM_GISEL_COV_PREFIX=",

        # TODO: These are configurable in gn, which means people might actually
        # want other values here.
        "HAVE_LIBEDIT=",
        "LLVM_ENABLE_TERMINFO=",
        "LLVM_ENABLE_LIBXML2=",
        "HAVE_MALLINFO2=",

        # Various Linux-only APIs.
        "HAVE_FUTIMENS=" + one_if_linux,
        "HAVE_LINK_H=" + one_if_linux,
        "HAVE_LSEEK64=" + one_if_linux,
        "HAVE_MALLINFO=" + one_if_linux,
        "HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC=" + one_if_linux,

        # Various Mac-only APIs.
        "HAVE_CRASHREPORTER_INFO=" + one_if_mac,
        "HAVE_DECL_ARC4RANDOM=" + one_if_mac,
        "HAVE_DLADDR=" + one_if_mac,
        "HAVE_MACH_MACH_H=" + one_if_mac,
        "HAVE_MALLOC_MALLOC_H=" + one_if_mac,
        "HAVE_MALLOC_ZONE_STATISTICS=" + one_if_mac,
        "HAVE_PROC_PID_RUSAGE=" + one_if_mac,
        "HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC=" + one_if_mac,
        "HAVE_UNW_ADD_DYNAMIC_FDE=" + one_if_mac,

        # Win-only APIs.
        "HAVE_DECL_STRERROR_S=" + one_if_win,
        "HAVE__CHSIZE_S=" + one_if_win,

        # General Posix defines.
        "HAVE_BACKTRACE=" + one_if_posix,
        "HAVE_POSIX_SPAWN=" + one_if_posix,
        "HAVE_PTHREAD_GETNAME_NP=" + one_if_posix,
        "HAVE_DEREGISTER_FRAME=" + one_if_posix,
        "HAVE_REGISTER_FRAME=" + one_if_posix,
        "HAVE_DLFCN_H=" + one_if_posix,
        "HAVE_DLOPEN=" + one_if_posix,
        "HAVE_FUTIMES=" + one_if_posix,
        "HAVE_GETPAGESIZE=" + one_if_posix,
        "HAVE_GETRLIMIT=" + one_if_posix,
        "HAVE_GETRUSAGE=" + one_if_posix,
        "HAVE_ISATTY=" + one_if_posix,
        "HAVE_LIBPTHREAD=" + one_if_posix,
        "HAVE_PTHREAD_SETNAME_NP=" + one_if_posix,
        "HAVE_PREAD=" + one_if_posix,
        "HAVE_PTHREAD_H=" + one_if_posix,
        "HAVE_PTHREAD_MUTEX_LOCK=" + one_if_posix,
        "HAVE_PTHREAD_RWLOCK_INIT=" + one_if_posix,
        "HAVE_SBRK=" + one_if_posix,
        "HAVE_SETENV=" + one_if_posix,
        "HAVE_SETRLIMIT=" + one_if_posix,
        "HAVE_SIGALTSTACK=" + one_if_posix,
        "HAVE_STRERROR_R=" + one_if_posix,
        "HAVE_SYSCONF=" + one_if_posix,
        "HAVE_SYS_IOCTL_H=" + one_if_posix,
        "HAVE_SYS_MMAN_H=" + one_if_posix,
        "HAVE_SYS_PARAM_H=" + one_if_posix,
        "HAVE_SYS_RESOURCE_H=" + one_if_posix,
        "HAVE_SYS_TIME_H=" + one_if_posix,
        "HAVE_TERMIOS_H=" + one_if_posix,
        "HAVE_UNISTD_H=" + one_if_posix,
        "HAVE__UNWIND_BACKTRACE=" + one_if_posix,

        # Miscellaneous corner case variables.
        "stricmp=" + maybe_value(is_win, "_stricmp"),
        "strdup=" + maybe_value(is_win, "_strdup"),
        "LTDL_SHLIB_EXT=" + shlib_ext,
        "LLVM_PLUGIN_EXT=" + shlib_ext,
        "BACKTRACE_HEADER=" + maybe_value(is_posix, "execinfo.h"),

        # This is oddly duplicated with llvm-config.h.
        "LLVM_DEFAULT_TARGET_TRIPLE=" + triple,
        "LLVM_SUPPORT_XCODE_SIGNPOSTS=",
    ]

def get_llvm_config_values(host_os, arch, triple):
    is_win = host_os == "win"
    is_posix = not is_win
    return [
        "LLVM_BUILD_LLVM_DYLIB=",
        "LLVM_BUILD_SHARED_LIBS=",
        "LLVM_DEFAULT_TARGET_TRIPLE=" + triple,
        "LLVM_ENABLE_DUMP=",
        "LLVM_FORCE_ENABLE_STATS=",
        "LLVM_FORCE_USE_OLD_TOOLCHAIN=",
        "LLVM_HAS_ATOMICS=1",
        "LLVM_HAVE_TF_API=",
        "LLVM_HOST_TRIPLE=" + triple,
        "LLVM_NATIVE_ARCH=" + arch,
        "LLVM_UNREACHABLE_OPTIMIZE=1",
        "LLVM_USE_INTEL_JITEVENTS=",
        "LLVM_USE_OPROFILE=",
        "LLVM_USE_PERF=",
        "LLVM_WITH_Z3=",

        # TODO: Define this properly.
        "LLVM_VERSION_MAJOR=15",
        "LLVM_VERSION_MINOR=0",
        "LLVM_VERSION_PATCH=0",
        "PACKAGE_VERSION=15.0.0git",
        "LLVM_ON_UNIX=" + maybe_one(is_posix),
        "HAVE_SYSEXITS_H=" + maybe_one(is_posix),
        "LLVM_NATIVE_ASMPARSER=LLVMInitialize{}AsmParser".format(arch),
        "LLVM_NATIVE_ASMPRINTER=LLVMInitialize{}AsmPrinter".format(arch),
        "LLVM_NATIVE_DISASSEMBLER=LLVMInitialize{}Disassembler".format(arch),
        "LLVM_NATIVE_TARGET=LLVMInitialize{}Target".format(arch),
        "LLVM_NATIVE_TARGETINFO=LLVMInitialize{}TargetInfo".format(arch),
        "LLVM_NATIVE_TARGETMC=LLVMInitialize{}TargetMC".format(arch),
        "LLVM_NATIVE_TARGETMCA=LLVMInitialize{}TargetMCA".format(arch),

        # TODO: These are configurable in gn, which means people might actually
        # want other values here.
        "LLVM_ENABLE_THREADS=1",
        "LLVM_HAVE_LIBXAR=",
        "LLVM_ENABLE_ZLIB=",
        "LLVM_ENABLE_CURL=",
        "LLVM_ENABLE_DIA_SDK=",
    ]

# TODO: We should split out host vs. target here.
# TODO: Figure out how to use select so that we can share this translation from
# Bazel platform configuration to LLVM host and target selection.
config_h_values = select({
    "@bazel_tools//src/conditions:windows": get_config_values("win", "X86", "x86_64-pc-win32"),
    "@bazel_tools//src/conditions:darwin_arm64": get_config_values("mac", "AArch64", "arm64-apple-darwin"),
    "@bazel_tools//src/conditions:darwin_x86_64": get_config_values("mac", "X86", "x86_64-unknown-darwin"),
    "@bazel_tools//src/conditions:linux_aarch64": get_config_values("linux", "AArch64", "aarch64-unknown-linux-gnu"),
    "@bazel_tools//src/conditions:linux_ppc64le": get_config_values("linux", "PowerPC", "powerpc64le-unknown-linux-gnu"),
    "@bazel_tools//src/conditions:linux_s390x": get_config_values("linux", "SystemZ", "systemz-unknown-linux_gnu"),
    "//conditions:default": get_config_values("linux", "X86", "x86_64-unknown-linux-gnu"),
})

llvm_config_h_values = select({
    "@bazel_tools//src/conditions:windows": get_llvm_config_values("win", "X86", "x86_64-pc-win32"),
    "@bazel_tools//src/conditions:darwin_arm64": get_llvm_config_values("mac", "AArch64", "arm64-apple-darwin"),
    "@bazel_tools//src/conditions:darwin_x86_64": get_llvm_config_values("mac", "X86", "x86_64-unknown-darwin"),
    "@bazel_tools//src/conditions:linux_aarch64": get_llvm_config_values("linux", "AArch64", "aarch64-unknown-linux-gnu"),
    "@bazel_tools//src/conditions:linux_ppc64le": get_llvm_config_values("linux", "PowerPC", "powerpc64le-unknown-linux-gnu"),
    "@bazel_tools//src/conditions:linux_s390x": get_llvm_config_values("linux", "SystemZ", "systemz-unknown-linux_gnu"),
    "//conditions:default": get_llvm_config_values("linux", "X86", "x86_64-unknown-linux-gnu"),
})

linux_defines = [
    "_GNU_SOURCE",
]

win32_defines = [
    # Windows system library specific defines.
    "_CRT_SECURE_NO_DEPRECATE",
    "_CRT_SECURE_NO_WARNINGS",
    "_CRT_NONSTDC_NO_DEPRECATE",
    "_CRT_NONSTDC_NO_WARNINGS",
    "_SCL_SECURE_NO_DEPRECATE",
    "_SCL_SECURE_NO_WARNINGS",
    "UNICODE",
    "_UNICODE",
]

# TODO: We should switch to platforms-based config settings to make this easier
# to express.
os_defines = select({
    "@bazel_tools//src/conditions:windows": win32_defines,
    "@bazel_tools//src/conditions:darwin": [],
    "@bazel_tools//src/conditions:freebsd": [],
    "//conditions:default": linux_defines,
})

# These shouldn't be needed by the C++11 standard, but are for some
# platforms (e.g. glibc < 2.18. See
# https://sourceware.org/bugzilla/show_bug.cgi?id=15366). These are also
# included unconditionally in the CMake build:
# https://github.com/llvm/llvm-project/blob/cd0dd8ece8e/llvm/cmake/modules/HandleLLVMOptions.cmake#L907-L909
llvm_global_defines = os_defines + [
    "__STDC_LIMIT_MACROS",
    "__STDC_CONSTANT_MACROS",
    "__STDC_FORMAT_MACROS",
]
