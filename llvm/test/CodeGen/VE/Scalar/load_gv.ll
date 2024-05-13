; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

@vi8 = common dso_local local_unnamed_addr global i8 0, align 1
@vi16 = common dso_local local_unnamed_addr global i16 0, align 2
@vi32 = common dso_local local_unnamed_addr global i32 0, align 4
@vi64 = common dso_local local_unnamed_addr global i64 0, align 8
@vi128 = common dso_local local_unnamed_addr global i128 0, align 16
@vf32 = common dso_local local_unnamed_addr global float 0.000000e+00, align 4
@vf64 = common dso_local local_unnamed_addr global double 0.000000e+00, align 8
@vf128 = common dso_local local_unnamed_addr global fp128 0xL00000000000000000000000000000000, align 16

; Function Attrs: norecurse nounwind readonly
define fp128 @loadf128com() {
; CHECK-LABEL: loadf128com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vf128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, vf128@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load fp128, ptr @vf128, align 16
  ret fp128 %1
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
  %1 = load double, ptr @vf64, align 8
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
  %1 = load float, ptr @vf32, align 4
  ret float %1
}

; Function Attrs: norecurse nounwind readonly
define i128 @loadi128com() {
; CHECK-LABEL: loadi128com:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, vi128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, vi128@hi(, %s0)
; CHECK-NEXT:    ld %s0, (, %s1)
; CHECK-NEXT:    ld %s1, 8(, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %1 = load i128, ptr @vi128, align 16
  ret i128 %1
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
  %1 = load i64, ptr @vi64, align 8
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
  %1 = load i32, ptr @vi32, align 4
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
  %1 = load i16, ptr @vi16, align 2
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
  %1 = load i8, ptr @vi8, align 1
  ret i8 %1
}
