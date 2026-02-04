; RUN: llc -mtriple s390x-zos -emit-gnuas-syntax-on-zos=0 < %s | FileCheck %s

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
; CHECK: stdin#C CSECT
; CHECK: C_WSA64 CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(stdi
; CHECK-NEXT:                in#S)
; CHECK: stdin#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)

; Check that there is no call to sqrt.
; CHECK-NOT:  DC   RD(@@WSQT@B)
; CHECK-NOT:  DC   VD(@@WSQT@B)

; Check that there is the correct library call.
; CHECK:      DC   RD(@@FCOS@B)
; CHECK-NEXT: DC   VD(@@FCOS@B)
; CHECK:      DC   RD(@@SSIN@B)
; CHECK-NEXT: DC   VD(@@SSIN@B)
; CHECK:      DC   RD(@@LXP2@B)
; CHECK-NEXT: DC   VD(@@LXP2@B)
