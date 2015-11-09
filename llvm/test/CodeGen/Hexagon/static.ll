; RUN: llc -march=hexagon -mcpu=hexagonv4 -disable-dfa-sched -disable-hexagon-misched < %s | FileCheck %s
; XFAIL: *

@num = external global i32
@acc = external global i32
@val = external global i32

; CHECK: memw(##num)
; CHECK: memw(##acc)
; CHECK: memw(##val)

define void @foo() nounwind {
entry:
  %0 = load i32, i32* @num, align 4
  %1 = load i32, i32* @acc, align 4
  %mul = mul nsw i32 %0, %1
  %2 = load i32, i32* @val, align 4
  %add = add nsw i32 %mul, %2
  store i32 %add, i32* @num, align 4
  ret void
}
