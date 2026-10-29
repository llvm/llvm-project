; RUN: llc -mtriple=riscv32 -mcpu=veer-eh1 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=VEER
; RUN: llc -mtriple=riscv32 -mtune=veer-eh1 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=VEER
; RUN: llc -mtriple=riscv32 -mcpu=generic-rv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=GENERIC

; VeeR EH1 raises the minimum jump-table size to 10 entries: a dense 9-case
; switch lowers to a branch tree, a 10-case switch still uses a jump table.

define void @nine_cases(i32 signext %in, ptr %out) nounwind {
; VEER-LABEL: nine_cases:
; VEER-NOT:     .LJTI
; VEER:         ret
;
; GENERIC-LABEL: nine_cases:
; GENERIC:       .LJTI0_0
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
    i32 7, label %bb7
    i32 8, label %bb8
    i32 9, label %bb9
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
bb5:
  store i32 100, ptr %out
  br label %exit
bb6:
  store i32 200, ptr %out
  br label %exit
bb7:
  store i32 300, ptr %out
  br label %exit
bb8:
  store i32 400, ptr %out
  br label %exit
bb9:
  store i32 500, ptr %out
  br label %exit
exit:
  ret void
}

define void @ten_cases(i32 signext %in, ptr %out) nounwind {
; VEER-LABEL: ten_cases:
; VEER:         .LJTI1_0
;
; GENERIC-LABEL: ten_cases:
; GENERIC:       .LJTI1_0
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
    i32 7, label %bb7
    i32 8, label %bb8
    i32 9, label %bb9
    i32 10, label %bb10
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
bb5:
  store i32 100, ptr %out
  br label %exit
bb6:
  store i32 200, ptr %out
  br label %exit
bb7:
  store i32 300, ptr %out
  br label %exit
bb8:
  store i32 400, ptr %out
  br label %exit
bb9:
  store i32 500, ptr %out
  br label %exit
bb10:
  store i32 600, ptr %out
  br label %exit
exit:
  ret void
}
