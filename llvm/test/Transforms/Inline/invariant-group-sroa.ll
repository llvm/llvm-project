; RUN: opt -passes='print<inline-cost>' -disable-output %s 2>&1 | FileCheck %s

; SROA analysis should yield non-zero savings for allocas passed through invariant group intrinsics
; CHECK: SROACostSavings: 10

declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)

declare void @b()

define i32 @f() {
  %a = alloca i32
  %r = call i32 @g(ptr %a)
  ret i32 %r
}

define i32 @g(ptr %a) {
  %a_inv_i8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %i1 = load i32, ptr %a_inv_i8
  %i2 = load i32, ptr %a_inv_i8
  %i3 = add i32 %i1, %i2
  %t = call ptr @llvm.strip.invariant.group.p0(ptr %a_inv_i8)
  ret i32 %i3
}
