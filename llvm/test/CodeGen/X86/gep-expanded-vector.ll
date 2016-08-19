; RUN: llc < %s -O2 -mattr=avx512f -mtriple=x86_64-unknown | FileCheck %s

%struct.S1 = type { %struct.S1*, %struct.S1* }

define %struct.S1** @malloc_init_state(<64 x %struct.S1**> %tmp, i32 %ind) {
entry:
  %Vec = getelementptr inbounds %struct.S1*, <64 x %struct.S1**> %tmp , i64 2
  %ptr = extractelement <64 x %struct.S1**> %Vec, i32 %ind
  ret %struct.S1** %ptr
}

; CHECK: .LCPI0_0:
; CHECK: .quad 16
; CHECK: vpbroadcastq    .LCPI0_0(%rip), [[Z1:%zmm[0-9]]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z2:%zmm[0-9]]], [[Z2]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z3:%zmm[0-9]]], [[Z3]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z4:%zmm[0-9]]], [[Z4]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z5:%zmm[0-9]]], [[Z5]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z6:%zmm[0-9]]], [[Z6]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z7:%zmm[0-9]]], [[Z7]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z8:%zmm[0-9]]], [[Z8]]
; CHECK-NEXT: vpaddq  [[Z1]], [[Z9:%zmm[0-9]]], [[Z9]]


