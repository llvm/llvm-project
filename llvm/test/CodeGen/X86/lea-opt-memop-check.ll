; REQUIRES: asserts
; RUN: llc < %s -march=x86 -mtriple=i686-pc-win32 -enable-x86-lea-opt | FileCheck %s

; PR26575
; Assertion `(Disp->isImm() || Disp->isGlobal()) && (Other.Disp->isImm() || Other.Disp->isGlobal()) && "Address displacement operand is always an immediate or a global"' failed.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) argmemonly nounwind
declare <2 x i64> @_mm_xor_si128(<2 x i64>, <2 x i64>) optsize
declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8) nounwind readnone
declare <4 x float> @_mm_castsi128_ps(<2 x i64>) optsize

define void @test(i8* nocapture readonly %src, i32 %len) #0 {
  %parts = alloca [4 x i32], align 4
  %part0 = bitcast [4 x i32]* %parts to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %part0, i8* %src, i32 %len, i32 1, i1 false)
  %call0 = tail call <2 x i64> @_mm_xor_si128(<2 x i64> undef, <2 x i64> <i64 -9187201950435737472, i64 -9187201950435737472>)
  %tmp0 = tail call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> undef, <2 x i64> <i64 7631803798, i64 5708721108>, i8 16)
  %call1 = tail call <4 x float> @_mm_castsi128_ps(<2 x i64> %tmp0)
  ret void
; CHECK-LABEL: test:
; CHECK:	leal{{.*}}
; CHECK:	calll _memcpy
; CHECK:	movaps __xmm@{{[0-9a-f]+}}, %xmm1
; CHECK:	calll __mm_xor_si128
; CHECK:	pclmulqdq $16, __xmm@{{[0-9a-f]+}}, %xmm0
; CHECK:	jmp __mm_castsi128_ps
}

attributes #0 = { nounwind optsize "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
