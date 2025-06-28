// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s 2>&1 | FileCheck -check-prefix=ERR %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def FUNC0 : RuntimeLibcall;
def FUNC1 : RuntimeLibcall;

def isFoo : RuntimeLibcallPredicate<[{isFoo()}]>;
def isBar : RuntimeLibcallPredicate<[{isBar()}]>;
def isTargetArch : RuntimeLibcallPredicate<[{isTargetArch()}]>;

def func0 : RuntimeLibcallImpl<FUNC0>;
def func1 : RuntimeLibcallImpl<FUNC1>;

// ERR: :[[@LINE+2]]:8: error: combining nested libcall set predicates currently unhandled
def TheSystemLibrary : SystemRuntimeLibrary<isTargetArch,
  (add LibcallImpls<(add func0, LibcallImpls<(add func1), isBar>), isFoo>)
>;
