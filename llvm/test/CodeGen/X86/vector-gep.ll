; RUN: llc < %s -march=x86 -mcpu=corei7-avx | FileCheck %s
; RUN: opt -instsimplify %s -disable-output

;CHECK: AGEP0:
define <4 x i32*> @AGEP0(i32* %ptr) nounwind {
entry:
  %vecinit.i = insertelement <4 x i32*> undef, i32* %ptr, i32 0
  %vecinit2.i = insertelement <4 x i32*> %vecinit.i, i32* %ptr, i32 1
  %vecinit4.i = insertelement <4 x i32*> %vecinit2.i, i32* %ptr, i32 2
  %vecinit6.i = insertelement <4 x i32*> %vecinit4.i, i32* %ptr, i32 3
;CHECK: pslld $2
;CHECK: padd
  %A2 = getelementptr <4 x i32*> %vecinit6.i, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
;CHECK: pslld $2
;CHECK: padd
  %A3 = getelementptr <4 x i32*> %A2, <4 x i32> <i32 10, i32 14, i32 19, i32 233>
  ret <4 x i32*> %A3
;CHECK: ret
}

;CHECK: AGEP1:
define i32 @AGEP1(<4 x i32*> %param) nounwind {
entry:
;CHECK: pslld $2
;CHECK: padd
  %A2 = getelementptr <4 x i32*> %param, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  %k = extractelement <4 x i32*> %A2, i32 3
  %v = load i32* %k
  ret i32 %v
;CHECK: ret
}

;CHECK: AGEP2:
define i32 @AGEP2(<4 x i32*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK: pslld $2
;CHECK: padd
  %A2 = getelementptr <4 x i32*> %param, <4 x i32> %off
  %k = extractelement <4 x i32*> %A2, i32 3
  %v = load i32* %k
  ret i32 %v
;CHECK: ret
}

;CHECK: AGEP3:
define <4 x i32*> @AGEP3(<4 x i32*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK: pslld $2
;CHECK: padd
  %A2 = getelementptr <4 x i32*> %param, <4 x i32> %off
  %v = alloca i32
  %k = insertelement <4 x i32*> %A2, i32* %v, i32 3
  ret <4 x i32*> %k
;CHECK: ret
}

;CHECK: AGEP4:
define <4 x i16*> @AGEP4(<4 x i16*> %param, <4 x i32> %off) nounwind {
entry:
; Multiply offset by two (add it to itself).
;CHECK: padd
; add the base to the offset
;CHECK: padd
  %A = getelementptr <4 x i16*> %param, <4 x i32> %off
  ret <4 x i16*> %A
;CHECK: ret
}

;CHECK: AGEP5:
define <4 x i8*> @AGEP5(<4 x i8*> %param, <4 x i8> %off) nounwind {
entry:
;CHECK: paddd
  %A = getelementptr <4 x i8*> %param, <4 x i8> %off
  ret <4 x i8*> %A
;CHECK: ret
}


; The size of each element is 1 byte. No need to multiply by element size.
;CHECK: AGEP6:
define <4 x i8*> @AGEP6(<4 x i8*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK-NOT: pslld
  %A = getelementptr <4 x i8*> %param, <4 x i32> %off
  ret <4 x i8*> %A
;CHECK: ret
}

