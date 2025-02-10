# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Substitutions for use in templates related to the LLVM configuration."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "//:vars.bzl",
    "LLVM_VERSION_MAJOR",
    "LLVM_VERSION_MINOR",
    "LLVM_VERSION_PATCH",
    "PACKAGE_VERSION",
)
load("//llvm:targets.bzl", "llvm_targets")
load(
    "//config:cmakehelpers.bzl",
    "cmakedefine",
    "cmakedefine01",
    "cmakedefine01_off",
    "cmakedefine01_on",
    "cmakedefine_set",
    "cmakedefine_sset",
    "cmakedefine_sunset",
    "cmakedefine_unset",
    "cmakedefine_vset",
    "cmakedefine_vunset",
)

CONFIG_H_SUBSTITUTIONS = (
    {"${BUG_REPORT_URL}": "https://github.com/llvm/llvm-project/issues/"} |
    cmakedefine01(
        "ENABLE_BACKTRACES",
        disable = "//config:LLVM_ENABLE_BACKTRACES_disabled",
    ) |
    cmakedefine01(
        "ENABLE_CRASH_OVERRIDES",
        disable = "//config:LLVM_ENABLE_CRASH_OVERRIDES_disabled",
    ) |
    cmakedefine01(
        "LLVM_ENABLE_CRASH_DUMPS",
        enable = "//config:LLVM_ENABLE_CRASH_DUMPS_enabled",
    ) |
    cmakedefine01(
        "ENABLE_DEBUGLOC_COVERAGE_TRACKING",
        enable = "//config:LLVM_ENABLE_DEBUGLOC_COVERAGE_TRACKING_coverage",
    ) |
    cmakedefine01(
        "LLVM_WINDOWS_PREFER_FORWARD_SLASH",
        enable = (
            "@bazel_tools//tools/cpp:mingw",
            "//config:LLVM_WINDOWS_PREFER_FORWARD_SLASH_enabled",
        ),
        disable = (
            "//config:LLVM_WINDOWS_PREFER_FORWARD_SLASH_disabled",
            "//conditions:default",
        ),
    ) |
    cmakedefine("HAVE_BACKTRACE", enable = "//config:posix") |
    {"${BACKTRACE_HEADER}": "execinfo.h"} |
    cmakedefine_unset("HAVE_CRASHREPORTERCLIENT_H") |
    cmakedefine01_off("HAVE_CRASHREPORTER_INFO") |
    cmakedefine01_off("HAVE_DECL_ARC4RANDOM") |
    cmakedefine01_on("HAVE_DECL_FE_ALL_EXCEPT") |
    cmakedefine01_on("HAVE_DECL_FE_INEXACT") |
    cmakedefine01_off("HAVE_DECL_STRERROR_S") |
    cmakedefine_vset("HAVE_DLOPEN") |
    cmakedefine("HAVE_REGISTER_FRAME", enable = "//config:posix") |
    cmakedefine("HAVE_DEREGISTER_FRAME", enable = "//config:posix") |
    cmakedefine("HAVE_UNW_ADD_DYNAMIC_FDE", enable = "@platforms//os:macos") |
    cmakedefine_vunset("HAVE_FFI_CALL") |
    cmakedefine_vunset("HAVE_FFI_FFI_H") |
    cmakedefine_vunset("HAVE_FFI_H") |
    cmakedefine_vset("HAVE_FUTIMENS") |
    cmakedefine_vset("HAVE_FUTIMES") |
    cmakedefine_vset("HAVE_GETPAGESIZE") |
    cmakedefine_vset("HAVE_GETRUSAGE") |
    {"#cmakedefine HAVE_ISATTY 1": "#define HAVE_ISATTY 1"} |
    cmakedefine_vunset("HAVE_LIBEDIT") |
    # TODO: Wire this up Exegesis and remove custom logic.
    cmakedefine_vunset("HAVE_LIBPFM") |
    cmakedefine_vunset("LIBPFM_HAS_FIELD_CYCLES") |
    cmakedefine_vunset("HAVE_LIBPSAPI") |
    cmakedefine("HAVE_LIBPTHREAD", enable = "//config:posix") |
    cmakedefine("HAVE_PTHREAD_GETNAME_NP", enable = "//config:posix") |
    cmakedefine("HAVE_PTHREAD_SETNAME_NP", enable = "//config:posix") |
    cmakedefine_vunset("HAVE_PTHREAD_GET_NAME_NP") |  # TODO: Likely wrong?
    cmakedefine_vunset("HAVE_PTHREAD_SET_NAME_NP") |  # TODO: Likely wrong?
    cmakedefine("HAVE_MACH_MACH_H", enable = "@platforms//os:macos") |
    cmakedefine_vunset("HAVE_MALLCTL") |
    cmakedefine("HAVE_MALLINFO", enable = "@platforms//os:linux") |
    # TODO(aaronmondal): Make configurable and enable by default.
    cmakedefine_vunset("HAVE_MALLINFO2") |
    cmakedefine("HAVE_MALLOC_MALLOC_H", enable = "@platforms//os:macos") |
    cmakedefine(
        "HAVE_MALLOC_ZONE_STATISTICS",
        enable = "@platforms//os:macos",
    ) |
    cmakedefine_vset("HAVE_POSIX_SPAWN") |
    cmakedefine_vset("HAVE_PREAD") |
    cmakedefine("HAVE_PTHREAD_H", enable = "//config:posix") |
    cmakedefine_vset("HAVE_PTHREAD_MUTEX_LOCK") |
    cmakedefine_vset("HAVE_PTHREAD_RWLOCK_INIT") |
    cmakedefine("HAVE_SBRK", enable = "@platforms//os:linux") |
    cmakedefine("HAVE_SETENV", enable = "//config:posix") |
    cmakedefine_vset("HAVE_SIGALTSTACK") |
    cmakedefine("HAVE_STRERROR_R", enable = "//config:posix") |
    cmakedefine_vset("HAVE_SYSCONF") |
    cmakedefine_vset("HAVE_SYS_MMAN_H") |
    cmakedefine_vunset("HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC") |
    cmakedefine(
        "HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC",
        enable = "@platforms//os:linux",
    ) |
    cmakedefine("HAVE_UNISTD_H", enable = "//config:posix") |
    cmakedefine_vunset("HAVE_VALGRIND_VALGRIND_H") |
    cmakedefine_vunset("HAVE__ALLOCA") |
    cmakedefine_vunset("HAVE__CHSIZE_S") |
    cmakedefine_vset("HAVE__UNWIND_BACKTRACE") |
    cmakedefine_vunset("HAVE___ALLOCA") |
    cmakedefine_vunset("HAVE___ASHLDI3") |
    cmakedefine_vunset("HAVE___ASHRDI3") |
    cmakedefine_vunset("HAVE___CHKSTK") |
    cmakedefine_vunset("HAVE___CHKSTK_MS") |
    cmakedefine_vunset("HAVE___CMPDI2") |
    cmakedefine_vunset("HAVE___DIVDI3") |
    cmakedefine_vunset("HAVE___FIXDFDI") |
    cmakedefine_vunset("HAVE___FIXSFDI") |
    cmakedefine_vunset("HAVE___FLOATDIDF") |
    cmakedefine_vunset("HAVE___LSHRDI3") |
    cmakedefine_vunset("HAVE___MAIN") |
    cmakedefine_vunset("HAVE___MODDI3") |
    cmakedefine_vunset("HAVE___UDIVDI3") |
    cmakedefine_vunset("HAVE___UMODDI3") |
    cmakedefine_vunset("HAVE____CHKSTK") |
    cmakedefine_vunset("HAVE____CHKSTK_MS") |
    cmakedefine_sunset("HOST_LINK_VERSION") |
    cmakedefine_sunset("LLVM_TARGET_TRIPLE_ENV") |
    cmakedefine01_on("LLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO") |
    cmakedefine01_on("LLVM_VERSION_PRINTER_SHOW_BUILD_CONFIG") |
    cmakedefine_vunset("LLVM_ENABLE_LIBXML2") |
    select({
        "@platforms//os:windows": cmakedefine_sset("LTDL_SHLIB_EXT", ".dll"),
        "@platforms//os:macos": cmakedefine_sset("LTDL_SHLIB_EXT", ".dylib"),
        "//conditions:default": cmakedefine_sset("LTDL_SHLIB_EXT", ".so"),
    }) |
    select({
        "@platforms//os:windows": cmakedefine_sset("LLVM_PLUGIN_EXT", ".dll"),
        "@platforms//os:macos": cmakedefine_sset("LLVM_PLUGIN_EXT", ".dylib"),
        "//conditions:default": cmakedefine_sset("LLVM_PLUGIN_EXT", ".so"),
    }) |
    cmakedefine_sset(
        "PACKAGE_BUGREPORT",
        "https://github.com/llvm/llvm-project/issues/",
    ) |
    cmakedefine_sset("PACKAGE_NAME", "LLVM") |
    cmakedefine_sset(
        "PACKAGE_STRING",
        'PACKAGE_NAME " " LLVM_VERSION_STRING',
    ) |
    cmakedefine_sset("PACKAGE_VERSION", "LLVM_VERSION_STRING") |
    cmakedefine_sunset("PACKAGE_VENDOR") |
    select({
        "@platforms//os:windows": cmakedefine_sset("stricmp", "_stricmp"),
        "//conditions:default": cmakedefine_vunset("stricmp"),
    }) |
    select({
        "@platforms//os:windows": cmakedefine_sset("strdup", "_strdup"),
        "//conditions:default": cmakedefine_vunset("strdup"),
    }) |
    cmakedefine01_off("LLVM_GISEL_COV_ENABLED") |
    cmakedefine_sunset("LLVM_GISEL_COV_PREFIX") |
    cmakedefine01_off("LLVM_SUPPORT_XCODE_SIGNPOSTS") |
    select({
        "@platforms//os:macos": {
            "#cmakedefine HAVE_PROC_PID_RUSAGE 1": "#define HAVE_PROC_PID_RUSAGE 1",
        },
        "//conditions:default": {
            "#cmakedefine HAVE_PROC_PID_RUSAGE 1": "/* undef HAVE_PROC_PID_RUSAGE */",
        },
    }) |
    cmakedefine(
        "HAVE_BUILTIN_THREAD_POINTER",
        enable = "@platforms//os:linux",
        disable = (
            # HAVE_BUILTIN_THREAD_POINTER is true for on Linux outside of ppc64
            # for all recent toolchains.
            "//config:powerpc64le-unknown-linux-gnu",
            "//conditions:default",
        ),
    ) |
    cmakedefine_vset("HAVE_GETAUXVAL")
)

# TODO(aaronmondal): Consider reimplementing `@platforms` in a way that
#                    bidirectionally maps to llvm/TargetParser/Triple.h.
TRIPLES = [
    "aarch64-unknown-linux-gnu",
    "arm64-apple-darwin",
    "powerpc64le-unknown-linux-gnu",
    "systemz-unknown-linux_gnu",
    "x86_64-pc-win32",
    "x86_64-unknown-darwin",
    "x86_64-unknown-linux-gnu",
]

LLVM_CONFIG_H_SUBSTITUTIONS = (
    cmakedefine_unset("LLVM_ENABLE_DUMP") |
    # TODO(aaronmondal): Make this independent of the host triple.
    select({
        "//config:{}".format(triple): {"${LLVM_DEFAULT_TARGET_TRIPLE}": triple}
        for triple in TRIPLES
    }) |
    cmakedefine01_on("LLVM_ENABLE_THREADS") |
    cmakedefine01_on("LLVM_HAS_ATOMICS") |
    select({
        "//config:{}".format(triple): cmakedefine_sset(
            "LLVM_HOST_TRIPLE",
            triple,
        )
        for triple in TRIPLES
    }) |
    select({
        "@platforms//cpu:{}".format(cpu): {
            "#cmakedefine LLVM_NATIVE_ARCH ${LLVM_NATIVE_ARCH}": "#define LLVM_NATIVE_ARCH {}".format(arch),
            "#cmakedefine LLVM_NATIVE_ASMPARSER LLVMInitialize${LLVM_NATIVE_ARCH}AsmParser": "#define LLVM_NATIVE_ASMPARSER LLVMInitialize{}AsmParser".format(arch),
            "#cmakedefine LLVM_NATIVE_ASMPRINTER LLVMInitialize${LLVM_NATIVE_ARCH}AsmPrinter": "#define LLVM_NATIVE_ASMPRINTER LLVMInitialize{}AsmPrinter".format(arch),
            "#cmakedefine LLVM_NATIVE_DISASSEMBLER LLVMInitialize${LLVM_NATIVE_ARCH}Disassembler": "#define LLVM_NATIVE_DISASSEMBLER LLVMInitialize{}Disassembler".format(arch),
            "#cmakedefine LLVM_NATIVE_TARGET LLVMInitialize${LLVM_NATIVE_ARCH}Target": "#define LLVM_NATIVE_TARGET LLVMInitialize{}Target".format(arch),
            "#cmakedefine LLVM_NATIVE_TARGETINFO LLVMInitialize${LLVM_NATIVE_ARCH}TargetInfo": "#define LLVM_NATIVE_TARGETINFO LLVMInitialize{}TargetInfo".format(arch),
            "#cmakedefine LLVM_NATIVE_TARGETMC LLVMInitialize${LLVM_NATIVE_ARCH}TargetMC": "#define LLVM_NATIVE_TARGETMC LLVMInitialize{}TargetMC".format(arch),
            "#cmakedefine LLVM_NATIVE_TARGETMCA LLVMInitialize${LLVM_NATIVE_ARCH}TargetMCA": "#define LLVM_NATIVE_TARGETMCA LLVMInitialize{}TargetMCA".format(arch),
        }
        for (cpu, arch) in [
            ("x86_64", "X86"),
            ("aarch64", "AArch64"),
            ("ppc64le", "PowerPC"),
            ("s390x", "SystemZ"),
        ]
    }) |
    {
        "#cmakedefine01 LLVM_HAS_{}_TARGET".format(
            target,
        ): "#define LLVM_HAS_{}_TARGET {}".format(
            target,
            1 if target in llvm_targets else 0,
        )
        for target in [
            "AARCH64",
            "AMDGPU",
            "ARC",
            "ARM",
            "AVR",
            "BPF",
            "CSKY",
            "DIRECTX",
            "HEXAGON",
            "LANAI",
            "LOONGARCH",
            "M68K",
            "MIPS",
            "MSP430",
            "NVPTX",
            "POWERPC",
            "RISCV",
            "SPARC",
            "SPIRV",
            "SYSTEMZ",
            "VE",
            "WEBASSEMBLY",
            "X86",
            "XCORE",
            "XTENSA",
        ]
    } |
    cmakedefine("LLVM_ON_UNIX", enable = "//config:posix") |
    cmakedefine01_off("LLVM_USE_INTEL_JITEVENTS") |
    cmakedefine01_off("LLVM_USE_OPROFILE") |
    cmakedefine01_off("LLVM_USE_PERF") |
    {
        "${LLVM_VERSION_MAJOR}": LLVM_VERSION_MAJOR,
        "${LLVM_VERSION_MINOR}": LLVM_VERSION_MINOR,
        "${LLVM_VERSION_PATCH}": LLVM_VERSION_PATCH,
        "${PACKAGE_VERSION}": PACKAGE_VERSION,
    } |
    cmakedefine01_off("LLVM_FORCE_ENABLE_STATS") |
    cmakedefine_vunset("LLVM_WITH_Z3") |
    cmakedefine_vunset("LLVM_ENABLE_CURL") |
    cmakedefine_vunset("LLVM_ENABLE_HTTPLIB") |
    cmakedefine01(
        "LLVM_ENABLE_ZLIB",
        disable = "//config:LLVM_ENABLE_ZLIB_disabled",
    ) |
    cmakedefine01(
        "LLVM_ENABLE_ZSTD",
        disable = "//config:LLVM_ENABLE_ZSTD_disabled",
    ) |
    cmakedefine_unset("LLVM_HAVE_TFLITE") |
    cmakedefine("HAVE_SYSEXITS_H", enable = "//config:posix") |
    cmakedefine_unset("LLVM_BUILD_LLVM_DYLIB") |
    cmakedefine_unset("LLVM_BUILD_SHARED_LIBS") |
    cmakedefine_vunset("LLVM_FORCE_USE_OLD_TOOLCHAIN") |
    cmakedefine01_on("LLVM_UNREACHABLE_OPTIMIZE") |
    cmakedefine01_off("LLVM_ENABLE_DIA_SDK") |
    selects.with_or({
        # Note: Technically enabled for dynamic builds on Windows, but that's
        #       currently disabled globally in Bazel.
        (
            "@platforms//os:windows",
            "//config:LLVM_ENABLE_PLUGINS_disabled",
        ):
            cmakedefine_unset("LLVM_ENABLE_PLUGINS"),
        # TODO(aaronmondal): The CMake build seems to have a slightly different
        #                    intention for "OS390" but that doesn't seem to be a
        #                    valid CMake option. Fix or clarify this in CMake.
        (
            "//conditions:default",
            "//config:LLVM_ENABLE_PLUGINS_enabled",
        ):
            cmakedefine_set("LLVM_ENABLE_PLUGINS")
    }) |
    select({
        "//config:LLVM_HAS_LOGF128_enabled": cmakedefine_set("LLVM_HAS_LOGF128"),
        "//conditions:default": cmakedefine_unset("LLVM_HAS_LOGF128"),
    }) |
    cmakedefine_vunset("LLVM_BUILD_TELEMETRY")
)
