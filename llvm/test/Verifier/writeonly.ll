; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a() readnone writeonly
; CHECK: Attributes {{.*}} are incompatible

declare void @b() readonly writeonly
; CHECK: Attributes {{.*}} are incompatible

declare void @c(i32* readnone writeonly %p)
; CHECK: Attributes {{.*}} are incompatible

declare void @d(i32* readonly writeonly %p)
; CHECK: Attributes {{.*}} are incompatible
