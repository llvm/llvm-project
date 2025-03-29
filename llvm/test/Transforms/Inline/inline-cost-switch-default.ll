; RUN: opt -S -passes=inline %s -debug-only=inline-cost -min-jump-table-entries=4 --disable-output 2>&1 | FileCheck %s -check-prefix=LOOKUPTABLE -match-full-lines
; RUN: opt -S -passes=inline %s -debug-only=inline-cost -min-jump-table-entries=5 --disable-output 2>&1 | FileCheck %s -check-prefix=SWITCH -match-full-lines
; REQUIRES: x86_64-linux, asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @main(i64 %a) {
  %b = call i64 @small_switch_default(i64 %a)
  %c = call i64 @small_switch_no_default(i64 %a)
  %d = call i64 @lookup_table_default(i64 %a)
  %e = call i64 @lookup_table_no_default(i64 %a)
  ret i64 %b
}

; SWITCH-LABEL: Analyzing call of small_switch_default{{.*}}
; SWITCH: Cost: 0
define i64 @small_switch_default(i64 %a) {
  switch i64 %a, label %default_branch [
  i64 -1, label %branch_0
  i64 8, label %branch_1
  i64 52, label %branch_2
  ]

branch_0:
  br label %exit

branch_1:
  br label %exit

branch_2:
  br label %exit

default_branch:
  br label %exit

exit:
  %b = phi i64 [ 5, %branch_0 ], [ 9, %branch_1 ], [ 2, %branch_2 ], [ 3, %default_branch ]
  ret i64 %b
}

; SWITCH-LABEL: Analyzing call of small_switch_no_default{{.*}}
; SWITCH: Cost: -10
define i64 @small_switch_no_default(i64 %a) {
  switch i64 %a, label %unreachabledefault [
  i64 -1, label %branch_0
  i64 8, label %branch_1
  i64 52, label %branch_2
  ]

branch_0:
  br label %exit

branch_1:
  br label %exit

branch_2:
  br label %exit

unreachabledefault:
  unreachable

exit:
  %b = phi i64 [ 5, %branch_0 ], [ 9, %branch_1 ], [ 2, %branch_2 ]
  ret i64 %b
}

; LOOKUPTABLE-LABEL: Analyzing call of lookup_table_default{{.*}}
; LOOKUPTABLE: Cost: 10
; SWITCH-LABEL: Analyzing call of lookup_table_default{{.*}}
; SWITCH: Cost: 20
define i64 @lookup_table_default(i64 %a) {
  switch i64 %a, label %default_branch [
  i64 0, label %branch_0
  i64 1, label %branch_1
  i64 2, label %branch_2
  i64 3, label %branch_3
  ]

branch_0:
  br label %exit

branch_1:
  br label %exit

branch_2:
  br label %exit

branch_3:
  br label %exit

default_branch:
  br label %exit

exit:
  %b = phi i64 [ 5, %branch_0 ], [ 9, %branch_1 ], [ 2, %branch_2 ], [ 7, %branch_3 ], [ 3, %default_branch ]
  ret i64 %b
}

; LOOKUPTABLE-LABEL: Analyzing call of lookup_table_no_default{{.*}}
; LOOKUPTABLE: Cost: 0
; SWITCH-LABEL: Analyzing call of lookup_table_no_default{{.*}}
; SWITCH: Cost: 20
define i64 @lookup_table_no_default(i64 %a) {
  switch i64 %a, label %unreachabledefault [
  i64 0, label %branch_0
  i64 1, label %branch_1
  i64 2, label %branch_2
  i64 3, label %branch_3
  ]

branch_0:
  br label %exit

branch_1:
  br label %exit

branch_2:
  br label %exit

branch_3:
  br label %exit

unreachabledefault:
  unreachable

exit:
  %b = phi i64 [ 5, %branch_0 ], [ 9, %branch_1 ], [ 2, %branch_2 ], [ 7, %branch_3 ]
  ret i64 %b
}
