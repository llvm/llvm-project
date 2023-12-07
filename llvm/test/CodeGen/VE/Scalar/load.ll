; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readonly
define fp128 @loadf128(ptr nocapture readonly %0) {
; CHECK-LABEL: loadf128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s2, 8(, %s0)
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load fp128, ptr %0, align 16
  ret fp128 %2
}

; Function Attrs: norecurse nounwind readonly
define double @loadf64(ptr nocapture readonly %0) {
; CHECK-LABEL: loadf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load double, ptr %0, align 16
  ret double %2
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32(ptr nocapture readonly %0) {
; CHECK-LABEL: loadf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldu %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load float, ptr %0, align 16
  ret float %2
}

; Function Attrs: norecurse nounwind readonly
define i128 @loadi128(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s2, (, %s0)
; CHECK-NEXT:    ld %s1, 8(, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i128, ptr %0, align 16
  ret i128 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i64, ptr %0, align 16
  ret i64 %2
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i32, ptr %0, align 16
  ret i32 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi32sext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi32sext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i32, ptr %0, align 16
  %3 = sext i32 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi32zext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi32zext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i32, ptr %0, align 16
  %3 = zext i32 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i16, ptr %0, align 16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi16sext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi16sext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i16, ptr %0, align 16
  %3 = sext i16 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi16zext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi16zext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i16, ptr %0, align 16
  %3 = zext i16 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i8, ptr %0, align 16
  ret i8 %2
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi8sext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi8sext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i8, ptr %0, align 16
  %3 = sext i8 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi8zext(ptr nocapture readonly %0) {
; CHECK-LABEL: loadi8zext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = load i8, ptr %0, align 16
  %3 = zext i8 %2 to i64
  ret i64 %3
}

; Function Attrs: norecurse nounwind readonly
define fp128 @loadf128stk() {
; CHECK-LABEL: loadf128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s1, (, %s11)
; CHECK-NEXT:    ld %s0, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca fp128, align 16
  %1 = load fp128, ptr %addr, align 16
  ret fp128 %1
}

; Function Attrs: norecurse nounwind readonly
define double @loadf64stk() {
; CHECK-LABEL: loadf64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca double, align 16
  %1 = load double, ptr %addr, align 16
  ret double %1
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32stk() {
; CHECK-LABEL: loadf32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca float, align 16
  %1 = load float, ptr %addr, align 16
  ret float %1
}

; Function Attrs: norecurse nounwind readonly
define i128 @loadi128stk() {
; CHECK-LABEL: loadi128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s11)
; CHECK-NEXT:    ld %s1, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i128, align 16
  %1 = load i128, ptr %addr, align 16
  ret i128 %1
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64stk() {
; CHECK-LABEL: loadi64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i64, align 16
  %1 = load i64, ptr %addr, align 16
  ret i64 %1
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32stk() {
; CHECK-LABEL: loadi32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i32, align 16
  %1 = load i32, ptr %addr, align 16
  ret i32 %1
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16stk() {
; CHECK-LABEL: loadi16stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i16, align 16
  %1 = load i16, ptr %addr, align 16
  ret i16 %1
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8stk() {
; CHECK-LABEL: loadi8stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i8, align 16
  %1 = load i8, ptr %addr, align 16
  ret i8 %1
}
