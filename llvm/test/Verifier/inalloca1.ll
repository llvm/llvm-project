; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i64* byval(i64) inalloca %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @b(i64* inreg inalloca %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @c(i64* sret(i64) inalloca %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @d(i64* nest inalloca %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @e(i64* readonly inalloca %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @f(void ()* inalloca %p)
; CHECK: do not support unsized types

declare void @g(i32* inalloca %p, i32 %p2)
; CHECK: inalloca isn't on the last parameter!
