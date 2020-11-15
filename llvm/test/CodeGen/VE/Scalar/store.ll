; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readonly
define void @storef128(fp128* nocapture %0, fp128 %1) {
; CHECK-LABEL: storef128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s2, 8(, %s0)
; CHECK-NEXT:    st %s3, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store fp128 %1, fp128* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storef64(double* nocapture %0, double %1) {
; CHECK-LABEL: storef64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store double %1, double* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storef32(float* nocapture %0, float %1) {
; CHECK-LABEL: storef32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stu %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store float %1, float* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei128(i128* nocapture %0, i128 %1) {
; CHECK-LABEL: storei128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s2, 8(, %s0)
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i128 %1, i128* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei64(i64* nocapture %0, i64 %1) {
; CHECK-LABEL: storei64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i64 %1, i64* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei32(i32* nocapture %0, i32 %1) {
; CHECK-LABEL: storei32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i32 %1, i32* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei32tr(i32* nocapture %0, i64 %1) {
; CHECK-LABEL: storei32tr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = trunc i64 %1 to i32
  store i32 %3, i32* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei16(i16* nocapture %0, i16 %1) {
; CHECK-LABEL: storei16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i16 %1, i16* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei16tr(i16* nocapture %0, i64 %1) {
; CHECK-LABEL: storei16tr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = trunc i64 %1 to i16
  store i16 %3, i16* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei8(i8* nocapture %0, i8 %1) {
; CHECK-LABEL: storei8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i8 %1, i8* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei8tr(i8* nocapture %0, i64 %1) {
; CHECK-LABEL: storei8tr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = trunc i64 %1 to i8
  store i8 %3, i8* %0, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storef128stk(fp128 %0) {
; CHECK-LABEL: storef128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 176(, %s11)
; CHECK-NEXT:    st %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca fp128, align 16
  store fp128 %0, fp128* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storef64stk(double %0) {
; CHECK-LABEL: storef64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca double, align 16
  store double %0, double* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storef32stk(float %0) {
; CHECK-LABEL: storef32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    stu %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca float, align 16
  store float %0, float* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei128stk(i128 %0) {
; CHECK-LABEL: storei128stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 184(, %s11)
; CHECK-NEXT:    st %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i128, align 16
  store i128 %0, i128* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei64stk(i64 %0) {
; CHECK-LABEL: storei64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i64, align 16
  store i64 %0, i64* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei32stk(i32 %0) {
; CHECK-LABEL: storei32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    stl %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i32, align 16
  store i32 %0, i32* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei16stk(i16 %0) {
; CHECK-LABEL: storei16stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st2b %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i16, align 16
  store i16 %0, i16* %addr, align 16
  ret void
}

; Function Attrs: norecurse nounwind readonly
define void @storei8stk(i8 %0) {
; CHECK-LABEL: storei8stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st1b %s0, 176(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %addr = alloca i8, align 16
  store i8 %0, i8* %addr, align 16
  ret void
}
