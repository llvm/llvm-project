; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

@vi8 = common dso_local local_unnamed_addr global i8 0, align 1
@vi16 = common dso_local local_unnamed_addr global i16 0, align 1
@vi32 = common dso_local local_unnamed_addr global i32 0, align 1
@vi64 = common dso_local local_unnamed_addr global i64 0, align 1
@vf32 = common dso_local local_unnamed_addr global float 0.000000e+00, align 1
@vf64 = common dso_local local_unnamed_addr global double 0.000000e+00, align 1

; Function Attrs: norecurse nounwind readonly
define double @loadf64stk() {
; CHECK-LABEL: loadf64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca double, align 1
  %1 = load double, double* %addr, align 1
  ret double %1
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32stk() {
; CHECK-LABEL: loadf32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, 12(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca float, align 1
  %1 = load float, float* %addr, align 1
  ret float %1
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64stk() {
; CHECK-LABEL: loadi64stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i64, align 1
  %1 = load i64, i64* %addr, align 1
  ret i64 %1
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32stk() {
; CHECK-LABEL: loadi32stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 12(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i32, align 1
  %1 = load i32, i32* %addr, align 1
  ret i32 %1
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16stk() {
; CHECK-LABEL: loadi16stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, 14(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i16, align 1
  %1 = load i16, i16* %addr, align 1
  ret i16 %1
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8stk() {
; CHECK-LABEL: loadi8stk:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.zx %s0, 15(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %addr = alloca i8, align 1
  %1 = load i8, i8* %addr, align 1
  ret i8 %1
}

; Function Attrs: norecurse nounwind readonly
define double @loadf64com() {
; CHECK-LABEL: loadf64com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vf64@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vf64@hi(, %s0)
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load double, double* @vf64, align 1
  ret double %1
}

; Function Attrs: norecurse nounwind readonly
define float @loadf32com() {
; CHECK-LABEL: loadf32com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vf32@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vf32@hi(, %s0)
; CHECK-NEXT:    ldu %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load float, float* @vf32, align 1
  ret float %1
}

; Function Attrs: norecurse nounwind readonly
define i64 @loadi64com() {
; CHECK-LABEL: loadi64com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vi64@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vi64@hi(, %s0)
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load i64, i64* @vi64, align 1
  ret i64 %1
}

; Function Attrs: norecurse nounwind readonly
define i32 @loadi32com() {
; CHECK-LABEL: loadi32com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vi32@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vi32@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load i32, i32* @vi32, align 1
  ret i32 %1
}

; Function Attrs: norecurse nounwind readonly
define i16 @loadi16com() {
; CHECK-LABEL: loadi16com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vi16@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vi16@hi(, %s0)
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load i16, i16* @vi16, align 1
  ret i16 %1
}

; Function Attrs: norecurse nounwind readonly
define i8 @loadi8com() {
; CHECK-LABEL: loadi8com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vi8@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, vi8@hi(, %s0)
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load i8, i8* @vi8, align 1
  ret i8 %1
}

