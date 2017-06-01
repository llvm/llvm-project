; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx | FileCheck %s

; __float128 myFP128 = 1.0L;  // x86_64-linux-android
@myFP128 = global fp128 0xL00000000000000003FFF000000000000, align 16

; The first few parameters are passed in registers and the other are on stack.

define fp128 @TestParam_FP128_0(fp128 %d0, fp128 %d1, fp128 %d2, fp128 %d3, fp128 %d4, fp128 %d5, fp128 %d6, fp128 %d7, fp128 %d8, fp128 %d9, fp128 %d10, fp128 %d11, fp128 %d12, fp128 %d13, fp128 %d14, fp128 %d15, fp128 %d16, fp128 %d17, fp128 %d18, fp128 %d19) {
entry:
  ret fp128 %d0
; CHECK-LABEL: TestParam_FP128_0:
; CHECK-NOT:   mov
; CHECK:       retq
}

define fp128 @TestParam_FP128_1(fp128 %d0, fp128 %d1, fp128 %d2, fp128 %d3, fp128 %d4, fp128 %d5, fp128 %d6, fp128 %d7, fp128 %d8, fp128 %d9, fp128 %d10, fp128 %d11, fp128 %d12, fp128 %d13, fp128 %d14, fp128 %d15, fp128 %d16, fp128 %d17, fp128 %d18, fp128 %d19) {
entry:
  ret fp128 %d1
; CHECK-LABEL: TestParam_FP128_1:
; CHECK:       movaps  %xmm1, %xmm0
; CHECK-NEXT:  retq
}

define fp128 @TestParam_FP128_7(fp128 %d0, fp128 %d1, fp128 %d2, fp128 %d3, fp128 %d4, fp128 %d5, fp128 %d6, fp128 %d7, fp128 %d8, fp128 %d9, fp128 %d10, fp128 %d11, fp128 %d12, fp128 %d13, fp128 %d14, fp128 %d15, fp128 %d16, fp128 %d17, fp128 %d18, fp128 %d19) {
entry:
  ret fp128 %d7
; CHECK-LABEL: TestParam_FP128_7:
; CHECK:       movaps  %xmm7, %xmm0
; CHECK-NEXT:  retq
}

define fp128 @TestParam_FP128_8(fp128 %d0, fp128 %d1, fp128 %d2, fp128 %d3, fp128 %d4, fp128 %d5, fp128 %d6, fp128 %d7, fp128 %d8, fp128 %d9, fp128 %d10, fp128 %d11, fp128 %d12, fp128 %d13, fp128 %d14, fp128 %d15, fp128 %d16, fp128 %d17, fp128 %d18, fp128 %d19) {
entry:
  ret fp128 %d8
; CHECK-LABEL: TestParam_FP128_8:
; CHECK:       movaps 8(%rsp), %xmm0
; CHECK-NEXT:  retq
}

define fp128 @TestParam_FP128_9(fp128 %d0, fp128 %d1, fp128 %d2, fp128 %d3, fp128 %d4, fp128 %d5, fp128 %d6, fp128 %d7, fp128 %d8, fp128 %d9, fp128 %d10, fp128 %d11, fp128 %d12, fp128 %d13, fp128 %d14, fp128 %d15, fp128 %d16, fp128 %d17, fp128 %d18, fp128 %d19) {
entry:
  ret fp128 %d9
; CHECK-LABEL: TestParam_FP128_9:
; CHECK:       movaps 24(%rsp), %xmm0
; CHECK-NEXT:  retq
}
