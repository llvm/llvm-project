// Error cases for the composable predicate dag (all_of/any_of/not over
// LibcallPredicate leaves).

// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_NOT_ATOM %s 2>&1 | FileCheck -check-prefix=NOT-ATOM %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_BAD_OP %s 2>&1 | FileCheck -check-prefix=BAD-OP %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_NOT_ARITY %s 2>&1 | FileCheck -check-prefix=NOT-ARITY %s
// RUN: not llvm-tblgen -gen-runtime-libcalls -I %p/../../include -DERR_EMPTY %s 2>&1 | FileCheck -check-prefix=EMPTY %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def FUNC0 : RuntimeLibcall;
def impl0 : RuntimeLibcallImpl<FUNC0>;

def IsA : LibcallPredicate<[{TT.isAArch64()}]>;
def IsB : LibcallPredicate<[{TT.isOSDarwin()}]>;
def NotAnAtom : RuntimeLibcall;

def IsArch : LibcallPredicate<[{isArch()}]>;
def isArch : RuntimeLibcallAvailability<(all_of IsArch)>;

#ifdef ERR_NOT_ATOM
// NOT-ATOM: error: predicate dag leaf 'NotAnAtom' is not a LibcallPredicate
def badpred : RuntimeLibcallAvailability<(all_of NotAnAtom)>;
#endif

#ifdef ERR_BAD_OP
// A defined record that isn't all_of/any_of/not used as the dag operator.
// BAD-OP: error: unknown predicate dag operator 'IsA'; expected all_of/any_of/not
def badpred : RuntimeLibcallAvailability<(IsA IsB)>;
#endif

#ifdef ERR_NOT_ARITY
// NOT-ARITY: error: 'not' takes exactly one operand
def badpred : RuntimeLibcallAvailability<(not IsA, IsB)>;
#endif

#ifdef ERR_EMPTY
// EMPTY: error: 'all_of' requires at least one operand
def badpred : RuntimeLibcallAvailability<(all_of)>;
#endif

def Sys : SystemRuntimeLibrary<isArch,
  (add LibcallImpls<(add impl0), badpred>)>;
