// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DCONFLICT 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-CONFLICT


include "llvm/IR/Intrinsics.td"
// CHECK: foo = 1,
def int_foo : Intrinsic<[llvm_any_ty]>;

// No conflicts, since .bar is not a valid mangled type.
// CHECK: foo_bar,
def int_foo_bar : Intrinsic<[llvm_i32_ty]>;

// CHECK: foo_bar_f32,
def int_foo_bar_f32 : Intrinsic<[llvm_i32_ty]>;

// No conflicts, since i is not a valid mangled type without a bitwidth.
// CHECK: foo_i
def int_foo_i : Intrinsic<[llvm_i32_ty]>;

#ifdef CONFLICT
// CHECK-CONFLICT: error: intrinsic `llvm.foo.a3` cannot share prefix `llvm.foo.a3` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.bf16` cannot share prefix `llvm.foo.bf16` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.f32` cannot share prefix `llvm.foo.f32` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.f64` cannot share prefix `llvm.foo.f64` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.f_3` cannot share prefix `llvm.foo.f_3` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.i65` cannot share prefix `llvm.foo.i65` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.i65.bar` cannot share prefix `llvm.foo.i65` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.p3` cannot share prefix `llvm.foo.p3` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.s_3` cannot share prefix `llvm.foo.s_3` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: error: intrinsic `llvm.foo.v4` cannot share prefix `llvm.foo.v4` with another overloaded intrinsic `llvm.foo`
// CHECK-CONFLICT: 10 errors
// CHECK-CONFLICT-NOT: error

// Conflicts due to suffix being a possible mangled type.
def int_foo_f32 : Intrinsic<[llvm_i32_ty]>;
def int_foo_f64 : Intrinsic<[llvm_i32_ty]>;
def int_foo_bf16: Intrinsic<[llvm_i32_ty]>;
def int_foo_p3 : Intrinsic<[llvm_i32_ty]>;
def int_foo_a3 : Intrinsic<[llvm_i32_ty]>;
def int_foo_v4 : Intrinsic<[llvm_i32_ty]>;
def int_foo_func : Intrinsic<[llvm_i32_ty], [], [], "llvm.foo.f_3">;
def int_foo_struct : Intrinsic<[llvm_i32_ty], [], [], "llvm.foo.s_3">;
def int_foo_t3 : Intrinsic<[llvm_i32_ty]>;
def int_foo_i65 : Intrinsic<[llvm_i32_ty]>;
def int_foo_i65_bar : Intrinsic<[llvm_i32_ty]>;

#endif
