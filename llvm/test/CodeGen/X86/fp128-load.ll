; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx | FileCheck %s

; __float128 myFP128 = 1.0L;  // x86_64-linux-android
@my_fp128 = global fp128 0xL00000000000000003FFF000000000000, align 16

define fp128 @get_fp128() {
entry:
  %0 = load fp128, fp128* @my_fp128, align 16
  ret fp128 %0
; CHECK-LABEL: get_fp128:
; CHECK:       movaps my_fp128(%rip), %xmm0
; CHECK-NEXT:  retq
}

@TestLoadExtend.data = internal unnamed_addr constant [2 x float] [float 0x3FB99999A0000000, float 0x3FC99999A0000000], align 4

define fp128 @TestLoadExtend(fp128 %x, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [2 x float], [2 x float]* @TestLoadExtend.data, i64 0, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fpext float %0 to fp128
  ret fp128 %conv
; CHECK-LABEL: TestLoadExtend:
; CHECK:       movslq  %edi, %rax
; CHECK-NEXT:  movss   TestLoadExtend.data(,%rax,4), %xmm0
; CHECK-NEXT:  callq   __extendsftf2
; CHECK:       retq
}

; CHECK-LABEL:  my_fp128:
; CHECK-NEXT:  .quad   0
; CHECK-NEXT:  .quad   4611404543450677248
; CHECK-NEXT:  .size   my_fp128, 16
