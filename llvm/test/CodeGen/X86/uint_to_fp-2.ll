; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

; rdar://6504833
define float @test1(i32 %x) nounwind readnone {
; CHECK: test1
; CHECK: movd
; CHECK: orpd
; CHECK: subsd
; CHECK: cvtsd2ss
; CHECK: movss
; CHECK: flds
; CHECK: ret
entry:
	%0 = uitofp i32 %x to float
	ret float %0
}

; PR10802
define float @test2(<4 x i32> %x) nounwind readnone ssp {
; CHECK: test2
; CHECK: xorps [[ZERO:%xmm[0-9]+]]
; CHECK: movss {{.*}}, [[ZERO]]
; CHECK: orps
; CHECK: subsd
; CHECK: cvtsd2ss
; CHECK: movss
; CHECK: flds
; CHECK: ret
entry:
  %vecext = extractelement <4 x i32> %x, i32 0
  %conv = uitofp i32 %vecext to float
  ret float %conv
}
