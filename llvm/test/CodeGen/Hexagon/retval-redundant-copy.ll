; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
;
; Check whether there are no redundant register copies of return values
;
; CHECK: memw(gp+#g0) = r0
; CHECK: memw(gp+#g1) = r0

@g0 = external global i32
@g1 = external global i32

define void @f0() {
b0:
  %v0 = tail call i32 @f1(i32 1, i32 2, i32 3)
  store i32 %v0, ptr @g0, align 4
  %v1 = tail call i32 @f1(i32 4, i32 5, i32 6)
  store i32 %v1, ptr @g1, align 4
  ret void
}

declare i32 @f1(i32, i32, i32)
