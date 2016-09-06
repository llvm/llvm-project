; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl | FileCheck %s

; 256-bit

; CHECK-LABEL: vpandd256
; CHECK: vpandd %ymm
; CHECK: ret
define <8 x i32> @vpandd256(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = and <8 x i32> %a2, %b
  ret <8 x i32> %x
}

; CHECK-LABEL: vpandnd256
; CHECK: vpandnd %ymm
; CHECK: ret
define <8 x i32> @vpandnd256(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %b2 = xor <8 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %x = and <8 x i32> %a2, %b2
  ret <8 x i32> %x
}

; CHECK-LABEL: vpord256
; CHECK: vpord %ymm
; CHECK: ret
define <8 x i32> @vpord256(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = or <8 x i32> %a2, %b
  ret <8 x i32> %x
}

; CHECK-LABEL: vpxord256
; CHECK: vpxord %ymm
; CHECK: ret
define <8 x i32> @vpxord256(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <8 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %x = xor <8 x i32> %a2, %b
  ret <8 x i32> %x
}

; CHECK-LABEL: vpandq256
; CHECK: vpandq %ymm
; CHECK: ret
define <4 x i64> @vpandq256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = and <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK-LABEL: vpandnq256
; CHECK: vpandnq %ymm
; CHECK: ret
define <4 x i64> @vpandnq256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %b2 = xor <4 x i64> %b, <i64 -1, i64 -1, i64 -1, i64 -1>
  %x = and <4 x i64> %a2, %b2
  ret <4 x i64> %x
}

; CHECK-LABEL: vporq256
; CHECK: vporq %ymm
; CHECK: ret
define <4 x i64> @vporq256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = or <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK-LABEL: vpxorq256
; CHECK: vpxorq %ymm
; CHECK: ret
define <4 x i64> @vpxorq256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = xor <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; 128-bit

; CHECK-LABEL: vpandd128
; CHECK: vpandd %xmm
; CHECK: ret
define <4 x i32> @vpandd128(<4 x i32> %a, <4 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i32> %a, <i32 1, i32 1, i32 1, i32 1>
  %x = and <4 x i32> %a2, %b
  ret <4 x i32> %x
}

; CHECK-LABEL: vpandnd128
; CHECK: vpandnd %xmm
; CHECK: ret
define <4 x i32> @vpandnd128(<4 x i32> %a, <4 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i32> %a, <i32 1, i32 1, i32 1, i32 1>
  %b2 = xor <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %x = and <4 x i32> %a2, %b2
  ret <4 x i32> %x
}

; CHECK-LABEL: vpord128
; CHECK: vpord %xmm
; CHECK: ret
define <4 x i32> @vpord128(<4 x i32> %a, <4 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i32> %a, <i32 1, i32 1, i32 1, i32 1>
  %x = or <4 x i32> %a2, %b
  ret <4 x i32> %x
}

; CHECK-LABEL: vpxord128
; CHECK: vpxord %xmm
; CHECK: ret
define <4 x i32> @vpxord128(<4 x i32> %a, <4 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i32> %a, <i32 1, i32 1, i32 1, i32 1>
  %x = xor <4 x i32> %a2, %b
  ret <4 x i32> %x
}

; CHECK-LABEL: vpandq128
; CHECK: vpandq %xmm
; CHECK: ret
define <2 x i64> @vpandq128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = and <2 x i64> %a2, %b
  ret <2 x i64> %x
}

; CHECK-LABEL: vpandnq128
; CHECK: vpandnq %xmm
; CHECK: ret
define <2 x i64> @vpandnq128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %b2 = xor <2 x i64> %b, <i64 -1, i64 -1>
  %x = and <2 x i64> %a2, %b2
  ret <2 x i64> %x
}

; CHECK-LABEL: vporq128
; CHECK: vporq %xmm
; CHECK: ret
define <2 x i64> @vporq128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = or <2 x i64> %a2, %b
  ret <2 x i64> %x
}

; CHECK-LABEL: vpxorq128
; CHECK: vpxorq %xmm
; CHECK: ret
define <2 x i64> @vpxorq128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = xor <2 x i64> %a2, %b
  ret <2 x i64> %x
}
