; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readonly
define fp128 @loadf128(fp128* nocapture readonly %0) {
; CHECK-LABEL: loadf128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s2, 8(, %s0)
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load fp128, fp128* %0, align 16
  ret fp128 %2
}

; Function Attrs: norecurse nounwind readonly
define double @loadf64(double* nocapture readonly %0) {
; CHECK-LABEL: loadf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load double, double* %0, align 16
  ret double %2
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32(float* nocapture readonly %0) {
; CHECK-LABEL: loadf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load float, float* %0, align 16
  ret float %2
}

; Function Attrs: norecurse nounwind readonly
define i128 @loadi128(i128* nocapture readonly %0) {
; CHECK-LABEL: loadi128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s2, (, %s0)
; CHECK-NEXT:    ld %s1, 8(, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i128, i128* %0, align 16
  ret i128 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64(i64* nocapture readonly %0) {
; CHECK-LABEL: loadi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i64, i64* %0, align 16
  ret i64 %2
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32(i32* nocapture readonly %0) {
; CHECK-LABEL: loadi32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i32, i32* %0, align 16
  ret i32 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi32sext(i32* nocapture readonly %0) {
; CHECK-LABEL: loadi32sext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i32, i32* %0, align 16
  %3 = sext i32 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi32zext(i32* nocapture readonly %0) {
; CHECK-LABEL: loadi32zext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i32, i32* %0, align 16
  %3 = zext i32 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16(i16* nocapture readonly %0) {
; CHECK-LABEL: loadi16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i16, i16* %0, align 16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi16sext(i16* nocapture readonly %0) {
; CHECK-LABEL: loadi16sext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.sx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i16, i16* %0, align 16
  %3 = sext i16 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi16zext(i16* nocapture readonly %0) {
; CHECK-LABEL: loadi16zext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i16, i16* %0, align 16
  %3 = zext i16 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8(i8* nocapture readonly %0) {
; CHECK-LABEL: loadi8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i8, i8* %0, align 16
  ret i8 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi8sext(i8* nocapture readonly %0) {
; CHECK-LABEL: loadi8sext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i8, i8* %0, align 16
  %3 = sext i8 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi8zext(i8* nocapture readonly %0) {
; CHECK-LABEL: loadi8zext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = load i8, i8* %0, align 16
  %3 = zext i8 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define fp128 @loadf128stk() {
; CHECK-LABEL: loadf128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca fp128, align 16
  %1 = load fp128, fp128* %addr, align 16
  ret fp128 %1
}

; Function Attrs: norecurse nounwind readonly
define double @loadf64stk() {
; CHECK-LABEL: loadf64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca double, align 16
  %1 = load double, double* %addr, align 16
  ret double %1
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32stk() {
; CHECK-LABEL: loadf32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca float, align 16
  %1 = load float, float* %addr, align 16
  ret float %1
}

; Function Attrs: norecurse nounwind readonly
define i128 @loadi128stk() {
; CHECK-LABEL: loadi128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 176(, %s11)
; CHECK-NEXT:    ld %s1, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i128, align 16
  %1 = load i128, i128* %addr, align 16
  ret i128 %1
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64stk() {
; CHECK-LABEL: loadi64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i64, align 16
  %1 = load i64, i64* %addr, align 16
  ret i64 %1
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32stk() {
; CHECK-LABEL: loadi32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i32, align 16
  %1 = load i32, i32* %addr, align 16
  ret i32 %1
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16stk() {
; CHECK-LABEL: loadi16stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i16, align 16
  %1 = load i16, i16* %addr, align 16
  ret i16 %1
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8stk() {
; CHECK-LABEL: loadi8stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i8, align 16
  %1 = load i8, i8* %addr, align 16
  ret i8 %1
}
