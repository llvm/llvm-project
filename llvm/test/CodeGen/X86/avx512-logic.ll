; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: vpandd
; CHECK: vpandd %zmm
; CHECK: ret
define <16 x i32> @vpandd(<16 x i32> %a, <16 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <16 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
                            i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = and <16 x i32> %a2, %b
  ret <16 x i32> %x
}

; CHECK-LABEL: vpord
; CHECK: vpord %zmm
; CHECK: ret
define <16 x i32> @vpord(<16 x i32> %a, <16 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <16 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
                            i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = or <16 x i32> %a2, %b
  ret <16 x i32> %x
}

; CHECK-LABEL: vpxord
; CHECK: vpxord %zmm
; CHECK: ret
define <16 x i32> @vpxord(<16 x i32> %a, <16 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <16 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
                            i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = xor <16 x i32> %a2, %b
  ret <16 x i32> %x
}

; CHECK-LABEL: vpandq
; CHECK: vpandq %zmm
; CHECK: ret
define <8 x i64> @vpandq(<8 x i64> %a, <8 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i64> %a, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %x = and <8 x i64> %a2, %b
  ret <8 x i64> %x
}

; CHECK-LABEL: vporq
; CHECK: vporq %zmm
; CHECK: ret
define <8 x i64> @vporq(<8 x i64> %a, <8 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i64> %a, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %x = or <8 x i64> %a2, %b
  ret <8 x i64> %x
}

; CHECK-LABEL: vpxorq
; CHECK: vpxorq %zmm
; CHECK: ret
define <8 x i64> @vpxorq(<8 x i64> %a, <8 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i64> %a, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %x = xor <8 x i64> %a2, %b
  ret <8 x i64> %x
}


; CHECK-LABEL: orq_broadcast
; CHECK: vporq LCP{{.*}}(%rip){1to8}, %zmm0, %zmm0
; CHECK: ret
define <8 x i64> @orq_broadcast(<8 x i64> %a) nounwind {
  %b = or <8 x i64> %a, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  ret <8 x i64> %b
}

; CHECK-LABEL: andd512fold
; CHECK: vpandd (%
; CHECK: ret
define <16 x i32> @andd512fold(<16 x i32> %y, <16 x i32>* %x) {
entry:
  %a = load <16 x i32>, <16 x i32>* %x, align 4
  %b = and <16 x i32> %y, %a
  ret <16 x i32> %b
}

; CHECK-LABEL: andqbrst
; CHECK: vpandq  (%rdi){1to8}, %zmm
; CHECK: ret
define <8 x i64> @andqbrst(<8 x i64> %p1, i64* %ap) {
entry:
  %a = load i64, i64* %ap, align 8
  %b = insertelement <8 x i64> undef, i64 %a, i32 0
  %c = shufflevector <8 x i64> %b, <8 x i64> undef, <8 x i32> zeroinitializer
  %d = and <8 x i64> %p1, %c
  ret <8 x i64>%d
}
