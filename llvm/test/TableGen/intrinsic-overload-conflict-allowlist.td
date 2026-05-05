// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-PASS
// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DALLOWLIST 2>&1 | FileCheck %s --check-prefix=CHECK-ALLOWLIST
// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DALLOWLIST_EXPLICIT_NAME 2>&1 | FileCheck %s --check-prefix=CHECK-ALLOWLIST-EXPLICIT
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_ALLOWLISTED 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-ALLOWLISTED
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NONEXISTENT 2>&1 | FileCheck %s --check-prefix=CHECK-NONEXISTENT
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_OVERLOADED 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-OVERLOADED
// RUN: not llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DERROR_NOT_INTRINSIC 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INTRINSIC

/// Tests the `KnownOverloadConflict` allowlist for intrinsic prefix conflicts:
/// covers the pass paths (baseline + two allowlist forms) and four diagnostic
/// paths (unallowlisted conflict, nonexistent target, non-overloaded target,
/// and target that isn't an Intrinsic).

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

// === Baseline (always defined) ===========================================
// CHECK-PASS: foo
// CHECK-PASS-NOT: error:

def int_foo : Intrinsic<[llvm_any_ty]>;

// === Allowlisted conflict ================================================
#ifdef ALLOWLIST
// CHECK-ALLOWLIST: foo
// CHECK-ALLOWLIST-NOT: error:

let KnownOverloadConflict = "int_foo" in
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

#ifdef ALLOWLIST_EXPLICIT_NAME
// CHECK-ALLOWLIST-EXPLICIT: corge
// CHECK-ALLOWLIST-EXPLICIT-NOT: error:

def int_corge_dom : Intrinsic<[llvm_any_ty], [], [], "llvm.corge">;
let KnownOverloadConflict = "int_corge_dom" in
def int_corge_sub : Intrinsic<[llvm_i32_ty], [], [], "llvm.corge.f128">;
#endif

// === Conflict without allowlist ==========================================
#ifdef ERROR_NOT_ALLOWLISTED
// CHECK-NOT-ALLOWLISTED: error: intrinsic `llvm.foo.f128` cannot share prefix
// CHECK-NOT-ALLOWLISTED: note: if intentional, add `let KnownOverloadConflict = "int_foo" in`
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Allowlist names a nonexistent record ================================
#ifdef ERROR_NONEXISTENT
// CHECK-NONEXISTENT: error: intrinsic `int_foo_f128` has KnownOverloadConflict referencing unknown intrinsic `int_does_not_exist`
let KnownOverloadConflict = "int_does_not_exist" in
def int_foo_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Allowlist names a non-overloaded record =============================
#ifdef ERROR_NOT_OVERLOADED
// CHECK-NOT-OVERLOADED: error: intrinsic `int_baz_f128` has KnownOverloadConflict referencing `int_baz`, which is not an overloaded intrinsic
def int_baz : Intrinsic<[llvm_i32_ty]>;  // NOT overloaded (concrete return type)
let KnownOverloadConflict = "int_baz" in
def int_baz_f128 : Intrinsic<[llvm_i32_ty]>;
#endif

// === Allowlist names a record that exists but is not an Intrinsic ========
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
