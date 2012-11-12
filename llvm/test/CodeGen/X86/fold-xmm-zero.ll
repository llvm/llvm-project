; RUN: llc < %s -mtriple=i386-apple-macosx10.6.7 -mattr=+sse2 | FileCheck %s

; Simple test to make sure folding for special constants (like float zero)
; isn't completely broken.

; CHECK: divss	LCPI0

%0 = type { float, float, float, float, float, float, float, float }

define void @f() nounwind ssp {
entry:
  %0 = tail call %0 asm sideeffect "foo", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00) nounwind
  %asmresult = extractvalue %0 %0, 0
  %asmresult8 = extractvalue %0 %0, 1
  %asmresult9 = extractvalue %0 %0, 2
  %asmresult10 = extractvalue %0 %0, 3
  %asmresult11 = extractvalue %0 %0, 4
  %asmresult12 = extractvalue %0 %0, 5
  %asmresult13 = extractvalue %0 %0, 6
  %asmresult14 = extractvalue %0 %0, 7
  %div = fdiv float %asmresult, 0.000000e+00
  %1 = tail call %0 asm sideeffect "bar", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(float %div, float %asmresult8, float %asmresult9, float %asmresult10, float %asmresult11, float %asmresult12, float %asmresult13, float %asmresult14) nounwind
  %asmresult24 = extractvalue %0 %1, 0
  %asmresult25 = extractvalue %0 %1, 1
  %asmresult26 = extractvalue %0 %1, 2
  %asmresult27 = extractvalue %0 %1, 3
  %asmresult28 = extractvalue %0 %1, 4
  %asmresult29 = extractvalue %0 %1, 5
  %asmresult30 = extractvalue %0 %1, 6
  %asmresult31 = extractvalue %0 %1, 7
  %div33 = fdiv float %asmresult24, 0.000000e+00
  %2 = tail call %0 asm sideeffect "baz", "={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},0,1,2,3,4,5,6,7,~{dirflag},~{fpsr},~{flags}"(float %div33, float %asmresult25, float %asmresult26, float %asmresult27, float %asmresult28, float %asmresult29, float %asmresult30, float %asmresult31) nounwind
  ret void
}
