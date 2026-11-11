// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_SYSTEM_NAME %s 2>&1 | FileCheck -check-prefix=SYSTEM %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_EXCLUSION_NAME %s 2>&1 | FileCheck -check-prefix=EXCLUSION %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_LIBRARY_MEMBER %s 2>&1 | FileCheck -check-prefix=LIBRARY %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_FUNC : RuntimeLibcall;
def func_a : RuntimeLibcallImpl<SOME_FUNC>;

def IsTargetArchA : LibcallPredicate<[{isTargetArchA()}]>;
def isTargetArchA : RuntimeLibcallAvailability<(all_of IsTargetArchA)>;

#ifndef ERR_LIBRARY_MEMBER
def Libc : LibcallLibrary<"libc", (add func_a)>;
#endif

#ifdef ERR_SYSTEM_NAME
// SYSTEM: [[@LINE+2]]:5: error: SystemRuntimeLibrary default library `libm` is not the name of a LibcallLibrary
// SYSTEM-NEXT: def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, ["libc", "libm"]>;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, ["libc", "libm"]>;
#endif

#ifdef ERR_EXCLUSION_NAME
// EXCLUSION: [[@LINE+2]]:1: error: LibraryExclusion library `libm` is not the name of a LibcallLibrary
// EXCLUSION-NEXT: def : LibraryExclusion<"libm", isTargetArchA, [func_a]>;
def : LibraryExclusion<"libm", isTargetArchA, [func_a]>;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, ["libc"]>;
#endif

#ifdef ERR_LIBRARY_MEMBER
// LIBRARY: [[@LINE+4]]:5: error: entry for LibcallLibrary is not a RuntimeLibcallImpl
// LIBRARY-NEXT: def Libc : LibcallLibrary<"libc", (add SOME_FUNC)>;
// LIBRARY: note: invalid entry `SOME_FUNC`
// LIBRARY-NEXT: def SOME_FUNC : RuntimeLibcall;
def Libc : LibcallLibrary<"libc", (add SOME_FUNC)>;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, ["libc"]>;
#endif
