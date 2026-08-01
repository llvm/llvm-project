// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_FUNC : RuntimeLibcall;
def func_a : RuntimeLibcallImpl<SOME_FUNC>;

def IsTargetArchA : LibcallPredicate<[{isTargetArchA()}]>;
def isTargetArchA : RuntimeLibcallAvailability<(all_of IsTargetArchA)>;

// CHECK: [[@LINE+4]]:5: error: entry for SystemLibrary is not a RuntimeLibcallImpl
// CHECK-NEXT: def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA,
// CHECK: note: invalid entry `SOME_FUNC`
// CHECK-NEXT: def SOME_FUNC : RuntimeLibcall;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA,
  (add SOME_FUNC)
>;

