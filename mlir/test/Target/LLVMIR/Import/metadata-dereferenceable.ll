; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

define void @deref(i64 %0, ptr %1) {
  ; CHECK: llvm.inttoptr
  ; CHECK-SAME: dereferenceable<bytes = 4>
  %3 = inttoptr i64 %0 to ptr, !dereferenceable !0
  ; CHECK: llvm.load
  ; CHECK-SAME: dereferenceable<bytes = 8>
  %4 = load ptr, ptr %1, align 8, !dereferenceable !1
  ret void
}

define void @deref_or_null(i64 %0, ptr %1) {
  ; CHECK: llvm.inttoptr
  ; CHECK-SAME: dereferenceable<bytes = 4, mayBeNull = true>
  %3 = inttoptr i64 %0 to ptr, !dereferenceable_or_null !0
  ; CHECK: llvm.load
  ; CHECK-SAME: dereferenceable<bytes = 8, mayBeNull = true>
  %4 = load ptr, ptr %1, align 8, !dereferenceable_or_null !1
  ret void
}

!0 = !{i64 4}
!1 = !{i64 8}
