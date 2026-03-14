; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(ptr byval(i64) inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @b(ptr inreg inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @c(ptr sret(i64) inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @d(ptr nest inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @e(ptr readonly inalloca(i64) %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @f(ptr inalloca(void()) %p)
; CHECK: Attribute 'inalloca' does not support unsized types

declare void @g(ptr inalloca(i32) %p, i32 %p2)
; CHECK: inalloca isn't on the last parameter!

; CHECK: Attribute 'inalloca(i8)' applied to incompatible type!
; CHECK-NEXT: ptr @inalloca_not_pointer
define void @inalloca_not_pointer(i8 inalloca(i8)) {
  ret void
}
