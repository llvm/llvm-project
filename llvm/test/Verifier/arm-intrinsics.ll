; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

define void @f(i32* %p) {
; CHECK: Intrinsic requires elementtype attribute on first argument
  %a = call i32 @llvm.arm.ldrex.p0i32(i32* %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c = call i32 @llvm.arm.strex.p0i32(i32 0, i32* %p)

; CHECK: Intrinsic requires elementtype attribute on first argument
  %a2 = call i32 @llvm.arm.ldaex.p0i32(i32* %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c2 = call i32 @llvm.arm.stlex.p0i32(i32 0, i32* %p)
  ret void
}

declare i32 @llvm.arm.ldrex.p0i32(i32*)
declare i32 @llvm.arm.ldaex.p0i32(i32*)
declare i32 @llvm.arm.stlex.p0i32(i32, i32*)
declare i32 @llvm.arm.strex.p0i32(i32, i32*)