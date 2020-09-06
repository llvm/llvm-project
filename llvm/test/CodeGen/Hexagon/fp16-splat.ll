; RUN: llc -march=hexagon -enable-qfloat-codegen < %s | FileCheck %s
; REQUIRES: asserts

; Check that the splat of a constant, when it's used in a fp vector operation that
; needs to be expanded (such as 'fdiv' below), is handled properly.

; CHECK-LABEL: test1
; CHECK: [[VREG0:(v[0-9]+)]].h = vsplat(r0)
; CHECK: vmpy({{.*}}[[VREG0]].hf

define dllexport void @test1() local_unnamed_addr #0 {
entry:
  %0 = load half, half* undef, align 2
  %1 = fadd half 0xH0000, %0
  %2 = insertelement <2 x half> undef, half %1, i32 1
  %3 = shufflevector <2 x half> %2, <2 x half> undef, <64 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  %4 = fdiv <64 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, %3
  %5 = load i8*, i8** undef, align 4
  %6 = bitcast i8* %5 to half*
  %cgep109 = getelementptr half, half* %6, i32 undef
  %7 = bitcast half* %cgep109 to <64 x half>*
  %8 = fmul <64 x half> zeroinitializer, %4
  store <64 x half> %8, <64 x half>* %7, align 128
  ret void
}

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hmxv68,+hvx-qfloat" }
