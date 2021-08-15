; RUN: llc < %s -mtriple=i386-apple-macosx10.6.7 -mattr=+avx512fp16 -no-integrated-as | FileCheck %s

; Simple test to make sure folding for special constants (like half zero)
; isn't completely broken.

; CHECK: vdivsh LCPI0

%0 = type { half, half, half, half, half, half, half, half }

define void @f() nounwind ssp {
entry:
  %0 = tail call %0 asm sideeffect "foo", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(half 1.000000e+00, half 2.000000e+00, half 3.000000e+00, half 4.000000e+00, half 5.000000e+00, half 6.000000e+00, half 7.000000e+00, half 8.000000e+00) nounwind
  %asmresult = extractvalue %0 %0, 0
  %asmresult8 = extractvalue %0 %0, 1
  %asmresult9 = extractvalue %0 %0, 2
  %asmresult10 = extractvalue %0 %0, 3
  %asmresult11 = extractvalue %0 %0, 4
  %asmresult12 = extractvalue %0 %0, 5
  %asmresult13 = extractvalue %0 %0, 6
  %asmresult14 = extractvalue %0 %0, 7
  %div = fdiv half %asmresult, 0.000000e+00
  %1 = tail call %0 asm sideeffect "bar", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(half %div, half %asmresult8, half %asmresult9, half %asmresult10, half %asmresult11, half %asmresult12, half %asmresult13, half %asmresult14) nounwind
  %asmresult24 = extractvalue %0 %1, 0
  %asmresult25 = extractvalue %0 %1, 1
  %asmresult26 = extractvalue %0 %1, 2
  %asmresult27 = extractvalue %0 %1, 3
  %asmresult28 = extractvalue %0 %1, 4
  %asmresult29 = extractvalue %0 %1, 5
  %asmresult30 = extractvalue %0 %1, 6
  %asmresult31 = extractvalue %0 %1, 7
  %div33 = fdiv half %asmresult24, 0.000000e+00
  %2 = tail call %0 asm sideeffect "baz", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(half %div33, half %asmresult25, half %asmresult26, half %asmresult27, half %asmresult28, half %asmresult29, half %asmresult30, half %asmresult31) nounwind
  ret void
}
