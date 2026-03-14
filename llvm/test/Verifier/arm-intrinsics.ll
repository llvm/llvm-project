; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

define void @f(ptr %p) {
; CHECK: Intrinsic requires elementtype attribute on first argument
  %a = call i32 @llvm.arm.ldrex.p0(ptr %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c = call i32 @llvm.arm.strex.p0(i32 0, ptr %p)

; CHECK: Intrinsic requires elementtype attribute on first argument
  %a2 = call i32 @llvm.arm.ldaex.p0(ptr %p)
; CHECK: Intrinsic requires elementtype attribute on second argument
  %c2 = call i32 @llvm.arm.stlex.p0(i32 0, ptr %p)
  ret void
}

declare i32 @llvm.arm.ldrex.p0(ptr)
declare i32 @llvm.arm.ldaex.p0(ptr)
declare i32 @llvm.arm.stlex.p0(i32, ptr)
declare i32 @llvm.arm.strex.p0(i32, ptr)