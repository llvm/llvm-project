# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Defines variables that use selects to configure LLVM based on platform."""

def native_arch_defines(arch, triple):
    return [
        "LLVM_NATIVE_ARCH=\\\"{}\\\"".format(arch),
        "LLVM_NATIVE_ASMPARSER=LLVMInitialize{}AsmParser".format(arch),
        "LLVM_NATIVE_ASMPRINTER=LLVMInitialize{}AsmPrinter".format(arch),
        "LLVM_NATIVE_DISASSEMBLER=LLVMInitialize{}Disassembler".format(arch),
        "LLVM_NATIVE_TARGET=LLVMInitialize{}Target".format(arch),
        "LLVM_NATIVE_TARGETINFO=LLVMInitialize{}TargetInfo".format(arch),
        "LLVM_NATIVE_TARGETMC=LLVMInitialize{}TargetMC".format(arch),
        "LLVM_HOST_TRIPLE=\\\"{}\\\"".format(triple),
        "LLVM_DEFAULT_TARGET_TRIPLE=\\\"{}\\\"".format(triple),
    ]

posix_defines = [
    "LLVM_ON_UNIX=1",
    "HAVE_BACKTRACE=1",
    "BACKTRACE_HEADER=<execinfo.h>",
    "LTDL_SHLIB_EXT=\\\".so\\\"",
    "LLVM_PLUGIN_EXT=\\\".so\\\"",
    "LLVM_ENABLE_THREADS=1",
    "HAVE_SYSEXITS_H=1",
    "HAVE_UNISTD_H=1",
    "HAVE_STRERROR_R=1",
    "HAVE_LIBPTHREAD=1",
    "HAVE_PTHREAD_GETNAME_NP=1",
    "HAVE_PTHREAD_SETNAME_NP=1",
    "HAVE_PTHREAD_GETSPECIFIC=1",
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
]

win32_defines = [
    # MSVC specific
    "stricmp=_stricmp",
    "strdup=_strdup",

    # LLVM features
    "LTDL_SHLIB_EXT=\\\".dll\\\"",
    "LLVM_PLUGIN_EXT=\\\".dll\\\"",
]

# TODO: We should switch to platforms-based config settings to make this easier
# to express.
os_defines = select({
    "@bazel_tools//src/conditions:windows": win32_defines,
    "@bazel_tools//src/conditions:darwin": macos_defines,
    "//conditions:default": linux_defines,
})

# TODO: We should split out host vs. target here.
llvm_config_defines = os_defines + select({
    "@bazel_tools//src/conditions:windows": native_arch_defines("X86", "x86_64-pc-win32"),
    "@bazel_tools//src/conditions:darwin": native_arch_defines("X86", "x86_64-unknown-darwin"),
    "@bazel_tools//src/conditions:linux_aarch64": native_arch_defines("AArch64", "aarch64-unknown-linux-gnu"),
    "//conditions:default": native_arch_defines("X86", "x86_64-unknown-linux-gnu"),
})
