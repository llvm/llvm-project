// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_SYSTEM_MEMBER %s 2>&1 | FileCheck -check-prefix=SYSTEM %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_LIBRARY_MEMBER %s 2>&1 | FileCheck -check-prefix=LIBRARY %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_FUNC : RuntimeLibcall;
def func_a : RuntimeLibcallImpl<SOME_FUNC>;

def IsTargetArchA : LibcallPredicate<[{isTargetArchA()}]>;
def isTargetArchA : RuntimeLibcallAvailability<(all_of IsTargetArchA)>;

#ifdef ERR_SYSTEM_MEMBER
// SYSTEM: [[@LINE+2]]:5: error: SystemRuntimeLibrary member `func_a` is not a LibcallLibrary or LibraryRef
// SYSTEM-NEXT: def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA,
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA,
  (add func_a)
>;
#endif

#ifdef ERR_LIBRARY_MEMBER
// LIBRARY: [[@LINE+4]]:5: error: entry for LibcallLibrary is not a RuntimeLibcallImpl
// LIBRARY-NEXT: def Libc : LibcallLibrary<"libc", (add SOME_FUNC)>;
// LIBRARY: note: invalid entry `SOME_FUNC`
// LIBRARY-NEXT: def SOME_FUNC : RuntimeLibcall;
def Libc : LibcallLibrary<"libc", (add SOME_FUNC)>;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, (add Libc)>;
#endif
