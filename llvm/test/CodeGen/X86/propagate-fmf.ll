; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx512fp16,avx512vl -debug 2>&1 | FileCheck %s

; Check that various fmf rewrites properly propagate the flags.

; FADD(acc, FMA(a, b, +0.0)) -> FMA(a, b, acc) 
define dso_local <32 x half> @test1(<32 x half> %acc, <32 x half> %a, <32 x half> %b) {
entry:
; CHECK:   Morphed node: [[T:t[0-9]+]]: v16f32 = VFCMADDCPHZr arcp contract
; CHECK-NOT: afn
; CHECK-NOT: nsz
  %0 = bitcast <32 x half> %a to <16 x float>
  %1 = bitcast <32 x half> %b to <16 x float>
  %2 = tail call nsz contract afn arcp <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float> %0, <16 x float> %1, <16 x float> zeroinitializer, i16 -1, i32 4)
  %3 = bitcast <16 x float> %2 to <32 x half>
  %add.i = fadd contract arcp <32 x half> %3, %acc
  ret <32 x half> %add.i
}

define double @test2(double %x, double %y) {
; CHECK:   Morphed node: [[T:t[0-9]+]]: v16f32 = VFCMADDCPHZr arcp contract
  %m = fmul fast double %x, %y
  %n = fneg fast double %m
  ret double %n
}

declare <16 x float> @llvm.x86.avx512fp16.mask.vfcmadd.cph.512(<16 x float>, <16 x float>, <16 x float>, i16, i32 immarg)

