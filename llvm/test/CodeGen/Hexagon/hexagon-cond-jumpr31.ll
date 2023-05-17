; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK: if (!p{{[0-3]}}.new) jumpr:nt r31
; CHECK-NOT: .falign

@g0 = common global i8 0, align 1
@g1 = common global i32 0, align 4

define i32 @f0(ptr nocapture %a0) {
b0:
  %v0 = load i8, ptr @g0, align 1
  %v1 = icmp eq i8 %v0, 65
  br i1 %v1, label %b1, label %b2

b1:                                               ; preds = %b0
  %v2 = load i32, ptr %a0, align 4
  %v3 = add nsw i32 %v2, 9
  %v4 = load i32, ptr @g1, align 4
  %v5 = sub i32 %v3, %v4
  store i32 %v5, ptr %a0, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret i32 undef
}
