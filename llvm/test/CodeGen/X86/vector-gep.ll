; RUN: llc < %s -mtriple=i686-linux -mcpu=corei7-avx | FileCheck %s
; RUN: opt -instsimplify -disable-output < %s

;CHECK-LABEL: AGEP0:
define <4 x i32*> @AGEP0(i32* %ptr) nounwind {
entry:
;CHECK-LABEL: AGEP0
;CHECK: vbroadcast
;CHECK-NEXT: vpaddd
;CHECK-NEXT: ret
  %vecinit.i = insertelement <4 x i32*> undef, i32* %ptr, i32 0
  %vecinit2.i = insertelement <4 x i32*> %vecinit.i, i32* %ptr, i32 1
  %vecinit4.i = insertelement <4 x i32*> %vecinit2.i, i32* %ptr, i32 2
  %vecinit6.i = insertelement <4 x i32*> %vecinit4.i, i32* %ptr, i32 3
  %A2 = getelementptr i32, <4 x i32*> %vecinit6.i, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  %A3 = getelementptr i32, <4 x i32*> %A2, <4 x i32> <i32 10, i32 14, i32 19, i32 233>
  ret <4 x i32*> %A3
}

;CHECK-LABEL: AGEP1:
define i32 @AGEP1(<4 x i32*> %param) nounwind {
entry:
;CHECK-LABEL: AGEP1
;CHECK: vpaddd
;CHECK-NEXT: vpextrd
;CHECK-NEXT: movl
  %A2 = getelementptr i32, <4 x i32*> %param, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  %k = extractelement <4 x i32*> %A2, i32 3
  %v = load i32, i32* %k
  ret i32 %v
;CHECK: ret
}

;CHECK-LABEL: AGEP2:
define i32 @AGEP2(<4 x i32*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK-LABEL: AGEP2
;CHECK: vpslld $2
;CHECK-NEXT: vpadd
  %A2 = getelementptr i32, <4 x i32*> %param, <4 x i32> %off
  %k = extractelement <4 x i32*> %A2, i32 3
  %v = load i32, i32* %k
  ret i32 %v
;CHECK: ret
}

;CHECK-LABEL: AGEP3:
define <4 x i32*> @AGEP3(<4 x i32*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK-LABEL: AGEP3
;CHECK: vpslld $2
;CHECK-NEXT: vpadd
  %A2 = getelementptr i32, <4 x i32*> %param, <4 x i32> %off
  %v = alloca i32
  %k = insertelement <4 x i32*> %A2, i32* %v, i32 3
  ret <4 x i32*> %k
;CHECK: ret
}

;CHECK-LABEL: AGEP4:
define <4 x i16*> @AGEP4(<4 x i16*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK-LABEL: AGEP4
; Multiply offset by two (add it to itself).
;CHECK: vpadd
; add the base to the offset
;CHECK-NEXT: vpadd
  %A = getelementptr i16, <4 x i16*> %param, <4 x i32> %off
  ret <4 x i16*> %A
;CHECK: ret
}

;CHECK-LABEL: AGEP5:
define <4 x i8*> @AGEP5(<4 x i8*> %param, <4 x i8> %off) nounwind {
entry:
;CHECK-LABEL: AGEP5
;CHECK: vpaddd
  %A = getelementptr i8, <4 x i8*> %param, <4 x i8> %off
  ret <4 x i8*> %A
;CHECK: ret
}


; The size of each element is 1 byte. No need to multiply by element size.
;CHECK-LABEL: AGEP6:
define <4 x i8*> @AGEP6(<4 x i8*> %param, <4 x i32> %off) nounwind {
entry:
;CHECK-LABEL: AGEP6
;CHECK-NOT: pslld
  %A = getelementptr i8, <4 x i8*> %param, <4 x i32> %off
  ret <4 x i8*> %A
;CHECK: ret
}

;CHECK-LABEL: AGEP7:
define <4 x i8*> @AGEP7(<4 x i8*> %param, i32 %off) nounwind {
entry:
;CHECK: vbroadcastss
;CHECK: vpadd
  %A = getelementptr i8, <4 x i8*> %param, i32 %off
  ret <4 x i8*> %A
;CHECK: ret
}

;CHECK-LABEL: AGEP8:
define <4 x i16*> @AGEP8(i16* %param, <4 x i32> %off) nounwind {
entry:
; Multiply offset by two (add it to itself).
;CHECK: vpadd
; add the base to the offset
;CHECK: vbroadcastss
;CHECK-NEXT: vpadd
  %A = getelementptr i16, i16* %param, <4 x i32> %off
  ret <4 x i16*> %A
;CHECK: ret
}

;CHECK-LABEL: AGEP9:
define <64 x i16*> @AGEP9(i16* %param, <64 x i32> %off) nounwind {
entry:
;CHECK: vbroadcastss
  %A = getelementptr i16, i16* %param, <64 x i32> %off
  ret <64 x i16*> %A
;CHECK: ret
}
