; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux-gnu -fast-isel -O0 < %s | FileCheck %s

define i1 @TestULT(double %t0) {
; CHECK-LABEL: TestULT:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp ult double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestULE(double %t0) {
; CHECK-LABEL: TestULE:
; CHECK: xscmpudp
; CHECK-NEXT: ble
; CHECK: blr
entry:
  %t1 = fcmp ule double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestUNE(double %t0) {
; CHECK-LABEL: TestUNE:
; CHECK: xscmpudp
; CHECK-NEXT: bne
; CHECK: blr
entry:
  %t1 = fcmp une double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestUEQ(double %t0) {
; CHECK-LABEL: TestUEQ:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp ueq double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestUGT(double %t0) {
; CHECK-LABEL: TestUGT:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp ugt double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestUGE(double %t0) {
; CHECK-LABEL: TestUGE:
; CHECK: xscmpudp
; CHECK-NEXT: bge
; CHECK: blr
entry:
  %t1 = fcmp uge double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestOLT(double %t0) {
; CHECK-LABEL: TestOLT:
; CHECK: xscmpudp
; CHECK-NEXT: blt
; CHECK: blr
entry:
  %t1 = fcmp olt double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestOLE(double %t0) {
; CHECK-LABEL: TestOLE:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp ole double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestONE(double %t0) {
; CHECK-LABEL: TestONE:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp one double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestOEQ(double %t0) {
; CHECK-LABEL: TestOEQ:
; CHECK: xscmpudp
; CHECK-NEXT: beq
; CHECK: blr
entry:
  %t1 = fcmp oeq double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestOGT(double %t0) {
; CHECK-LABEL: TestOGT:
; CHECK: xscmpudp
; CHECK-NEXT: bgt
; CHECK: blr
entry:
  %t1 = fcmp ogt double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}

define i1 @TestOGE(double %t0) {
; CHECK-LABEL: TestOGE:
; CHECK: xscmpudp
; CHECK: blr
entry:
  %t1 = fcmp oge double %t0, 0.000000e+00
  br i1 %t1, label %good, label %bad

bad:
  ret i1 false

good:
  ret i1 true
}
