; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=yonah | FileCheck %s
; RUN: llc < %s -march=x86-64 -mattr=+sse2 -mcpu=core2 | FileCheck %s

define i32 @t(<2 x i64>* %val) nounwind  {
; CHECK: t:
; CHECK-NOT: movd
; CHECK: movl 8(
; CHECK-NEXT: ret
	%tmp2 = load <2 x i64>* %val, align 16		; <<2 x i64>> [#uses=1]
	%tmp3 = bitcast <2 x i64> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp3, i32 2		; <i32> [#uses=1]
	ret i32 %tmp4
}

; Case where extractelement of load ends up as undef.
; (Making sure this doesn't crash.)
define i32 @t2(<8 x i32>* %xp) {
; CHECK: t2:
; CHECK: ret
  %x = load <8 x i32>* %xp
  %Shuff68 = shufflevector <8 x i32> %x, <8 x i32> undef, <8 x i32> <i32
undef, i32 7, i32 9, i32 undef, i32 13, i32 15, i32 1, i32 3>
  %y = extractelement <8 x i32> %Shuff68, i32 0
  ret i32 %y
}
