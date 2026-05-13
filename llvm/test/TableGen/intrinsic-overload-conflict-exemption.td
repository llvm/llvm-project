// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-PASS
// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DEXEMPTED 2>&1 | FileCheck %s --check-prefix=CHECK-EXEMPTED
// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DEXEMPTED_EXPLICIT_NAME 2>&1 | FileCheck %s --check-prefix=CHECK-EXEMPTED-EXPLICIT
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_EXEMPTED 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-EXEMPTED
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NONEXISTENT 2>&1 | FileCheck %s --check-prefix=CHECK-NONEXISTENT
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_OVERLOADED 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-OVERLOADED
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_INTRINSIC 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INTRINSIC

/// Tests the `KnownOverloadConflict` exemption for intrinsic prefix conflicts:
/// covers the pass paths (baseline + two exemption forms) and four diagnostic
/// paths (unexempted conflict, nonexistent target, non-overloaded target,
/// and target that isn't an Intrinsic).

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

// === Baseline (always defined) ===========================================
// CHECK-PASS: foo
// CHECK-PASS-NOT: error:

def int_foo : Intrinsic<[llvm_any_ty]>;

// === Exempted conflict ===================================================
#ifdef EXEMPTED
// CHECK-EXEMPTED: foo
// CHECK-EXEMPTED-NOT: error:

let KnownOverloadConflict = "int_foo" in
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

#ifdef EXEMPTED_EXPLICIT_NAME
// CHECK-EXEMPTED-EXPLICIT: corge
// CHECK-EXEMPTED-EXPLICIT-NOT: error:

def int_corge_dom : Intrinsic<[llvm_any_ty], [], [], "llvm.corge">;
let KnownOverloadConflict = "int_corge_dom" in
def int_corge_sub : Intrinsic<[llvm_i32_ty], [], [], "llvm.corge.f128">;
#endif

// === Conflict without exemption ==========================================
#ifdef ERROR_NOT_EXEMPTED
// CHECK-NOT-EXEMPTED: error: intrinsic `llvm.foo.f128` cannot share prefix
// CHECK-NOT-EXEMPTED: note: if intentional, set `KnownOverloadConflict = "int_foo"`
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Exemption names a nonexistent record ================================
#ifdef ERROR_NONEXISTENT
// CHECK-NONEXISTENT: error: intrinsic `int_foo_f128` has KnownOverloadConflict referencing unknown intrinsic `int_does_not_exist`
let KnownOverloadConflict = "int_does_not_exist" in
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Exemption names a non-overloaded record =============================
#ifdef ERROR_NOT_OVERLOADED
// CHECK-NOT-OVERLOADED: error: intrinsic `int_baz_f128` has KnownOverloadConflict referencing `int_baz`, which is not an overloaded intrinsic
def int_baz : Intrinsic<[llvm_i32_ty]>;  // NOT overloaded (concrete return type)
let KnownOverloadConflict = "int_baz" in
def int_baz_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Exemption names a record that exists but is not an Intrinsic ========
// Defends against the failure mode where a typo or stale rename points
// `KnownOverloadConflict` at some unrelated record (helper class, etc.).
// Without the `isSubClassOf("Intrinsic")` guard, `getValueAsDef("TypeInfo")`
// would `PrintFatalError` with a generic "no such field" message that
// fails to mention `KnownOverloadConflict` at all.
#ifdef ERROR_NOT_INTRINSIC
// CHECK-NOT-INTRINSIC: error: intrinsic `int_qux_f128` has KnownOverloadConflict referencing `some_helper`, which is not an Intrinsic
class Helper { string Whatever = ""; }
def some_helper : Helper;
let KnownOverloadConflict = "some_helper" in
def int_qux_f128 : Intrinsic<[llvm_i32_ty]>;
#endif
