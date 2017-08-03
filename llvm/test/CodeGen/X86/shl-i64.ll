; RUN: llc -march=x86 -mattr=+sse2 < %s | FileCheck %s

; Make sure that we don't generate an illegal i64 extract after LegalizeType.
; CHECK: shll


define void @test_cl(<4 x i64>*  %dst, <4 x i64>* %src, i32 %idx) {
entry:
  %arrayidx = getelementptr inbounds <4 x i64>, <4 x i64> * %src, i32 %idx
  %0 = load <4 x i64> , <4 x i64> * %arrayidx, align 32
  %arrayidx1 = getelementptr inbounds <4 x i64>, <4 x i64> * %dst, i32 %idx
  %1 = load <4 x i64> , <4 x i64> * %arrayidx1, align 32
  %2 = extractelement <4 x i64> %1, i32 0
  %and = and i64 %2, 63
  %3 = insertelement <4 x i64> undef, i64 %and, i32 0    
  %splat = shufflevector <4 x i64> %3, <4 x i64> undef, <4 x i32> zeroinitializer
  %shl = shl <4 x i64> %0, %splat
  store <4 x i64> %shl, <4 x i64> * %arrayidx1, align 32
  ret void
}
