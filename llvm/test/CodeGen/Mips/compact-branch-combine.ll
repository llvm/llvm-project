; RUN: llc -mtriple=mipsel -mcpu=mips32r6 < %s | FileCheck %s
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 < %s | FileCheck %s

;; Each function is a single compare + branch
;; Checking each pattern for compact branch is selected

; CHECK-NOT: \bslt
; CHECK-NOT: \bsltu
; CHECK-NOT: \bslti
; CHECK-NOT: \bsltiu


;; br + slt -> bgec
define void @test_slt(i32 %a, i32 %b) {
; CHECK-LABEL: test_slt:
; CHECK: bgec
  %c = icmp slt i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sgt -> bgec
define void @test_sgt(i32 %a, i32 %b) {
; CHECK-LABEL: test_sgt:
; CHECK: bgec
  %c = icmp sgt i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sge -> bgec/bltc
define void @test_sge(i32 %a, i32 %b) {
; CHECK-LABEL: test_sge:
; CHECK: {{bgec|bltc}}
  %c = icmp sge i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sle -> bgec/bltc
define void @test_sle(i32 %a, i32 %b) {
; CHECK-LABEL: test_sle:
; CHECK: {{bgec|bltc}}
  %c = icmp sle i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + ult -> bgeuc
define void @test_ult(i32 %a, i32 %b) {
; CHECK-LABEL: test_ult:
; CHECK: bgeuc
  %c = icmp ult i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + ugt -> bgeuc
define void @test_ugt(i32 %a, i32 %b) {
; CHECK-LABEL: test_ugt:
; CHECK: bgeuc
  %c = icmp ugt i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + uge -> bgeuc/bltuc
define void @test_uge(i32 %a, i32 %b) {
; CHECK-LABEL: test_uge:
; CHECK: {{bgeuc|bltuc}}
  %c = icmp uge i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + ule -> bgeuc/bltuc
define void @test_ule(i32 %a, i32 %b) {
; CHECK-LABEL: test_ule:
; CHECK: {{bgeuc|bltuc}}
  %c = icmp ule i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + slt rs,0 -> bltzc
define void @test_lt_zero(i32 %a) {
; CHECK-LABEL: test_lt_zero:
; CHECK: bltzc
  %c = icmp slt i32 %a, 0
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sgt rs,0 -> bgtzc/blezc
define void @test_gt_zero(i32 %a) {
; CHECK-LABEL: test_gt_zero:
; CHECK: {{bgtzc|blezc}}
  %c = icmp sgt i32 %a, 0
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sle rs,0 -> blezc/bgtzc
define void @test_le_zero(i32 %a) {
; CHECK-LABEL: test_le_zero:
; CHECK: {{blezc|bgtzc}}
  %c = icmp sle i32 %a, 0
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sge rs,0 -> bgezc/bltzc
define void @test_ge_zero(i32 %a) {
; CHECK-LABEL: test_ge_zero:
; CHECK: {{bgezc|bltzc}}
  %c = icmp sge i32 %a, 0
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + slt rs,1 -> blezc
define void @test_lt_one(i32 %a) {
; CHECK-LABEL: test_lt_one:
; CHECK: blezc
  %c = icmp slt i32 %a, 1
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sge rs,1 -> bgtzc/blezc
define void @test_ge_one(i32 %a) {
; CHECK-LABEL: test_ge_one:
; CHECK: {{bgtzc|blezc}}
  %c = icmp sge i32 %a, 1
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sgt rs,-1 -> bgezc/bltzc
define void @test_gt_minus1(i32 %a) {
; CHECK-LABEL: test_gt_minus1:
; CHECK: {{bgezc|bltzc}}
  %c = icmp sgt i32 %a, -1
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

;; br + sle rs,-1 -> bltzc/bgezc
define void @test_le_minus1(i32 %a) {
; CHECK-LABEL: test_le_minus1:
; CHECK: {{bltzc|bgezc}}
  %c = icmp sle i32 %a, -1
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}
