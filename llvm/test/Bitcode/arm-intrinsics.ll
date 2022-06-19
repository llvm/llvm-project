; RUN: llvm-dis < %S/arm-intrinsics.bc | FileCheck %s

define void @f(i32* %p) {
; CHECK: call i32 @llvm.arm.ldrex.p0i32(i32* elementtype(i32)
  %a = call i32 @llvm.arm.ldrex.p0i32(i32* %p)
; CHECK: call i32 @llvm.arm.strex.p0i32(i32 0, i32* elementtype(i32)
  %c = call i32 @llvm.arm.strex.p0i32(i32 0, i32* %p)

; CHECK: call i32 @llvm.arm.ldaex.p0i32(i32* elementtype(i32)
  %a2 = call i32 @llvm.arm.ldaex.p0i32(i32* %p)
; CHECK: call i32 @llvm.arm.stlex.p0i32(i32 0, i32* elementtype(i32)
  %c2 = call i32 @llvm.arm.stlex.p0i32(i32 0, i32* %p)
  ret void
}

declare i32 @llvm.arm.ldrex.p0i32(i32*)
declare i32 @llvm.arm.ldaex.p0i32(i32*)
declare i32 @llvm.arm.stlex.p0i32(i32, i32*)
declare i32 @llvm.arm.strex.p0i32(i32, i32*)