// RUN: llvm-tblgen -gen-callingconv -I %p/../../include -I %p/Common %s | FileCheck %s

// Check that the CCIfThrows predicate class emits an ArgFlags.isThrows()
// guard, so that backends can implement a herbception-aware calling
// convention (return value plus a success/failure discriminant).

include "reg-with-subregs-common.td"

def RetCC_Throws : CallingConv<[
  CCIfThrows<CCCustom<"RetThrowsCC">>
]>;

// CHECK: if (ArgFlags.isThrows()) {
// CHECK:   if (RetThrowsCC(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
