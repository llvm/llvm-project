; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i64* byval(i64) inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @b(i64* inreg inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @c(i64* sret(i64) inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @d(i64* nest inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @e(i64* readonly inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @f(void ()* inalloca(void()) %p)
; CHECK: do not support unsized types

declare void @g(i32* inalloca(i32) %p, i32 %p2)
; CHECK: inalloca isn't on the last parameter!

; CHECK: Attribute 'inalloca' type does not match parameter!
; CHECK-NEXT: void (i32*)* @inalloca_mismatched_pointee_type0
define void @inalloca_mismatched_pointee_type0(i32* inalloca(i8)) {
  ret void
}

; CHECK: Wrong types for attribute:
; CHECK-NEXT: void (i8)* @inalloca_not_pointer
define void @inalloca_not_pointer(i8 byref(i8)) {
  ret void
}
