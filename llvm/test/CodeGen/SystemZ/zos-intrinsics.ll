; RUN: llc -mtriple s390x-zos < %s | FileCheck %s

define float @sqrt_ieee(float %x) {
entry:
  %res = call float @llvm.sqrt.f32(float %x)
  ret float %res
}

define float @cos_ieee(float %x) {
entry:
  %res = call float @llvm.cos.f32(float %x)
  ret float %res
}

define double @sin_ieee(double %x) {
entry:
  %res = call double @llvm.sin.f64(double %x)
  ret double %res
}

define fp128 @exp2_ieee(fp128 %x) {
entry:
  %res = call fp128 @llvm.exp2.f128(fp128 %x)
  ret fp128 %res
}

declare float @llvm.sqrt.f32(float)
declare float @llvm.cos.f32(float)
declare double @llvm.sin.f64(double)
declare fp128 @llvm.exp2.f128(fp128)

; Check the calls in the ADA.
; CHECK: .section ".ada"

; Check that there is no call to sqrt.
; CHECK-NOT:  .quad   R(@@WSQT@B)
; CHECK-NOT:  .quad   V(@@WSQT@B)

; Check that there is the correct library call.
; CHECK:      .quad   R(@@FCOS@B)
; CHECK-NEXT: .quad   V(@@FCOS@B)
; CHECK:      .quad   R(@@SSIN@B)
; CHECK-NEXT: .quad   V(@@SSIN@B)
; CHECK:      .quad   R(@@LXP2@B)
; CHECK-NEXT: .quad   V(@@LXP2@B)
