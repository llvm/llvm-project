; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

@num = external global i32
@acc = external global i32
@num2 = external global i32

; CHECK: allocframe
; CHECK: dealloc_return

define i32 @foo() nounwind {
entry:
  %i = alloca i32, align 4
  %0 = load i32* @num, align 4
  store i32 %0, i32* %i, align 4
  %1 = load i32* %i, align 4
  %2 = load i32* @acc, align 4
  %mul = mul nsw i32 %1, %2
  %3 = load i32* @num2, align 4
  %add = add nsw i32 %mul, %3
  store i32 %add, i32* %i, align 4
  %4 = load i32* %i, align 4
  ret i32 %4
}
