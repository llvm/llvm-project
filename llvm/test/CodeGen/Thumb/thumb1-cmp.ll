; RUN: llc < %s -mtriple thumbv6m-eabi | FileCheck %s

define i32 @slt_poweroftwo(i32 %a) {
; CHECK-LABEL: slt_poweroftwo:
; CHECK: .long   4095
  %b = icmp slt i32 %a, 4096
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @sle_poweroftwo(i32 %a) {
; CHECK-LABEL: sle_poweroftwo:
; CHECK: movs    r1, #1
; CHECK: lsls    r1, r1, #12
  %b = icmp sle i32 %a, 4096
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @sge_poweroftwo(i32 %a) {
; CHECK-LABEL: sge_poweroftwo:
; CHECK: movs    r1, #1
; CHECK: lsls    r1, r1, #12
  %b = icmp sge i32 %a, 4096
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @sgt_poweroftwo(i32 %a) {
; CHECK-LABEL: sgt_poweroftwo:
; CHECK: .long   4097
  %b = icmp sgt i32 %a, 4096
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @slt_nearpoweroftwo(i32 %a) {
; CHECK-LABEL: slt_nearpoweroftwo:
; CHECK: movs    r1, #1
; CHECK: lsls    r1, r1, #12
  %b = icmp slt i32 %a, 4097
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @sle_nearpoweroftwo(i32 %a) {
; CHECK-LABEL: sle_nearpoweroftwo:
; CHECK: .long   4095
  %b = icmp sle i32 %a, 4095
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}


define i32 @sge_nearpoweroftwo(i32 %a) {
; CHECK-LABEL: sge_nearpoweroftwo:
; CHECK: .long   4097
  %b = icmp sge i32 %a, 4097
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

define i32 @sgt_nearpoweroftwo(i32 %a) {
; CHECK-LABEL: sgt_nearpoweroftwo:
; CHECK: movs    r1, #1
; CHECK: lsls    r1, r1, #12
  %b = icmp sgt i32 %a, 4095
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}
