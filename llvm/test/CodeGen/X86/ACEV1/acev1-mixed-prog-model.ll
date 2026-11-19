; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 -o /dev/null 2>&1 | FileCheck %s

; A function cannot use both models, so report it, and report it only once.

; CHECK: in function mixed_models void (ptr): ACE tile operations mix the macro and the managed programming model
; CHECK-NOT: ACE tile operations mix the macro and the managed programming model
define void @mixed_models(ptr %p) nounwind {
  call void @llvm.x86.acev1.tilezero(i8 0)
  call void @llvm.x86.acev1.tilezero(i8 1)
  %t = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %v = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %t)
  store <256 x i32> %v, ptr %p
  %t2 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %v2 = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %t2)
  store <256 x i32> %v2, ptr %p
  ret void
}

declare void @llvm.x86.acev1.tilezero(i8 immarg)
declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx)
