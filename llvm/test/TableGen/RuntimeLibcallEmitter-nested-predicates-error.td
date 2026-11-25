// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s 2>&1 | FileCheck -check-prefix=ERR %s

// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DINVALID_MEMBER %s 2>&1 | FileCheck -check-prefix=INVALID %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def FUNC0 : RuntimeLibcall;
def FUNC1 : RuntimeLibcall;

def IsFoo : LibcallPredicate<[{isFoo()}]>;
def IsBar : LibcallPredicate<[{isBar()}]>;
def IsTargetArch : LibcallPredicate<[{isTargetArch()}]>;

def isFoo : RuntimeLibcallAvailability<(all_of IsFoo)>;
def isBar : RuntimeLibcallAvailability<(all_of IsBar)>;
def isTargetArch : RuntimeLibcallAvailability<(all_of IsTargetArch)>;

def func0 : RuntimeLibcallImpl<FUNC0>;
def func1 : RuntimeLibcallImpl<FUNC1>;

#ifdef INVALID_MEMBER
// INVALID: error: entry for LibcallLibrary is not a RuntimeLibcallImpl
// INVALID: note: invalid entry `FUNC1`
def Libc : LibcallLibrary<"libc",
  (add LibcallImpls<(add LibcallImpls<(add FUNC1), isBar>), isFoo>)
>;
#else
// ERR: :[[@LINE+2]]:8: error: combining nested libcall set predicates currently unhandled
def Libc : LibcallLibrary<"libc",
  (add LibcallImpls<(add func0, LibcallImpls<(add func1), isBar>), isFoo>)
>;

#endif

def TheSystemLibrary : SystemRuntimeLibrary<isTargetArch, ["libc"]>;
