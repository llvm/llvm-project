// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s
// XFAIL: vg_leak

include "llvm/IR/Intrinsics.td"

// Make sure we generate the long name without crashing
// CHECK: this_is_a_really_long_intrinsic_name_but_we_should_still_not_crash,  // llvm.this.is.a.really.long.intrinsic.name.but.we.should.still.not.crash
def int_foo : Intrinsic<[llvm_anyint_ty], [], [], "llvm.foo">;
def int_this_is_a_really_long_intrinsic_name_but_we_should_still_not_crash : Intrinsic<[llvm_anyint_ty], [], [], "llvm.this.is.a.really.long.intrinsic.name.but.we.should.still.not.crash">;

