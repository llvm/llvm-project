; RUN: opt -passes='print<inline-cost>' -disable-output %s 2>&1 | FileCheck %s

; SROA analysis should yield non-zero savings for allocas passed through invariant group intrinsics
; CHECK: SROACostSavings: 10

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)

declare void @b()

define i32 @f() {
  %a = alloca i32
  %r = call i32 @g(i32* %a)
  ret i32 %r
}

define i32 @g(i32* %a) {
  %a_i8 = bitcast i32* %a to i8*
  %a_inv_i8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a_i8)
  %a_inv = bitcast i8* %a_inv_i8 to i32*
  %i1 = load i32, i32* %a_inv
  %i2 = load i32, i32* %a_inv
  %i3 = add i32 %i1, %i2
  %t = call i8* @llvm.strip.invariant.group.p0i8(i8* %a_inv_i8)
  ret i32 %i3
}
