# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Defines variables that use selects to configure LLVM based on platform."""

def native_arch_defines(arch, triple):
    return [
        r'LLVM_NATIVE_ARCH=\"{}\"'.format(arch),
        "LLVM_NATIVE_ASMPARSER=LLVMInitialize{}AsmParser".format(arch),
        "LLVM_NATIVE_ASMPRINTER=LLVMInitialize{}AsmPrinter".format(arch),
        "LLVM_NATIVE_DISASSEMBLER=LLVMInitialize{}Disassembler".format(arch),
        "LLVM_NATIVE_TARGET=LLVMInitialize{}Target".format(arch),
        "LLVM_NATIVE_TARGETINFO=LLVMInitialize{}TargetInfo".format(arch),
        "LLVM_NATIVE_TARGETMC=LLVMInitialize{}TargetMC".format(arch),
        "LLVM_NATIVE_TARGETMCA=LLVMInitialize{}TargetMCA".format(arch),
        r'LLVM_HOST_TRIPLE=\"{}\"'.format(triple),
        r'LLVM_DEFAULT_TARGET_TRIPLE=\"{}\"'.format(triple),
    ]

posix_defines = [
    "LLVM_ON_UNIX=1",
    "HAVE_BACKTRACE=1",
    "BACKTRACE_HEADER=<execinfo.h>",
    r'LTDL_SHLIB_EXT=\".so\"',
    r'LLVM_PLUGIN_EXT=\".so\"',
    "LLVM_ENABLE_THREADS=1",
    "HAVE_DEREGISTER_FRAME=1",
    "HAVE_LIBPTHREAD=1",
    "HAVE_PTHREAD_GETNAME_NP=1",
    "HAVE_PTHREAD_GETSPECIFIC=1",
    "HAVE_PTHREAD_H=1",
    "HAVE_PTHREAD_SETNAME_NP=1",
    "HAVE_REGISTER_FRAME=1",
    "HAVE_SETENV_R=1",
    "HAVE_STRERROR_R=1",
    "HAVE_SYSEXITS_H=1",
    "HAVE_UNISTD_H=1",
]

linux_defines = posix_defines + [
    "_GNU_SOURCE",
    "HAVE_LINK_H=1",
    "HAVE_LSEEK64=1",
    "HAVE_MALLINFO=1",
    "HAVE_POSIX_FALLOCATE=1",
    "HAVE_SBRK=1",
    "HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC=1",
]

macos_defines = posix_defines + [
    "HAVE_MACH_MACH_H=1",
    "HAVE_MALLOC_MALLOC_H=1",
    "HAVE_MALLOC_ZONE_STATISTICS=1",
    "HAVE_PROC_PID_RUSAGE=1",
    "HAVE_UNW_ADD_DYNAMIC_FDE=1",
]

win32_defines = [
    # MSVC specific
    "stricmp=_stricmp",
    "strdup=_strdup",

    # LLVM features
    r'LTDL_SHLIB_EXT=\".dll\"',
    r'LLVM_PLUGIN_EXT=\".dll\"',
]

# TODO: We should switch to platforms-based config settings to make this easier
# to express.
os_defines = select({
    "@bazel_tools//src/conditions:windows": win32_defines,
    "@bazel_tools//src/conditions:darwin": macos_defines,
    "@bazel_tools//src/conditions:freebsd": posix_defines,
    "//conditions:default": linux_defines,
})

# TODO: We should split out host vs. target here.
llvm_config_defines = os_defines + select({
    "@bazel_tools//src/conditions:windows": native_arch_defines("X86", "x86_64-pc-win32"),
    "@bazel_tools//src/conditions:darwin_arm64": native_arch_defines("AArch64", "arm64-apple-darwin"),
    "@bazel_tools//src/conditions:darwin_x86_64": native_arch_defines("X86", "x86_64-unknown-darwin"),
    "@bazel_tools//src/conditions:linux_aarch64": native_arch_defines("AArch64", "aarch64-unknown-linux-gnu"),
    "//conditions:default": native_arch_defines("X86", "x86_64-unknown-linux-gnu"),
}) + [
    # These shouldn't be needed by the C++11 standard, but are for some
    # platforms (e.g. glibc < 2.18. See
    # https://sourceware.org/bugzilla/show_bug.cgi?id=15366). These are also
    # included unconditionally in the CMake build:
    # https://github.com/llvm/llvm-project/blob/cd0dd8ece8e/llvm/cmake/modules/HandleLLVMOptions.cmake#L907-L909
    "__STDC_LIMIT_MACROS",
    "__STDC_CONSTANT_MACROS",
    "__STDC_FORMAT_MACROS",
]
