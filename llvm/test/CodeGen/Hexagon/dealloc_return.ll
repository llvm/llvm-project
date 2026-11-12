; RUN: llc -mtriple=hexagon < %s | FileCheck %s

@g0 = external global i32
@g1 = external global i32
@g2 = external global i32

; CHECK: allocframe(r29,
; CHECK: dealloc_return
; CHECK-NEXT: }

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = load i32, ptr @g0, align 4
  store i32 %v1, ptr %v0, align 4
  %v2 = load i32, ptr %v0, align 4
  %v3 = load i32, ptr @g1, align 4
  %v4 = mul nsw i32 %v2, %v3
  %v5 = load i32, ptr @g2, align 4
  %v6 = add nsw i32 %v4, %v5
  store i32 %v6, ptr %v0, align 4
  %v7 = load i32, ptr %v0, align 4
  ret i32 %v7
}

attributes #0 = { nounwind "frame-pointer"="all" }
