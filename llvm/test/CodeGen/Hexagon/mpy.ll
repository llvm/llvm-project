; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: += mpyi

define void @foo(i32 %acc, i32 %num, i32 %num2) nounwind {
entry:
  %acc.addr = alloca i32, align 4
  %num.addr = alloca i32, align 4
  %num2.addr = alloca i32, align 4
  store i32 %acc, i32* %acc.addr, align 4
  store i32 %num, i32* %num.addr, align 4
  store i32 %num2, i32* %num2.addr, align 4
  %0 = load i32* %num.addr, align 4
  %1 = load i32* %acc.addr, align 4
  %mul = mul nsw i32 %0, %1
  %2 = load i32* %num2.addr, align 4
  %add = add nsw i32 %mul, %2
  store i32 %add, i32* %num.addr, align 4
  ret void
}
