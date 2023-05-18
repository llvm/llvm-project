; RUN: llc -mtriple=arm64-none-linux-gnu -mattr=-neon < %s | FileCheck %s

define float @copy_FPR32(float %a, float %b) {
;CHECK-LABEL: copy_FPR32:
;CHECK: fmov s0, s1
  ret float %b;
}
  
define double @copy_FPR64(double %a, double %b) {
;CHECK-LABEL: copy_FPR64:
;CHECK: fmov d0, d1
  ret double %b;
}
  
define fp128 @copy_FPR128(fp128 %a, fp128 %b) {
;CHECK-LABEL: copy_FPR128:
;CHECK: str	q1, [sp, #-16]!
;CHECK-NEXT: ldr	q0, [sp, #16]!
  ret fp128 %b;
}
