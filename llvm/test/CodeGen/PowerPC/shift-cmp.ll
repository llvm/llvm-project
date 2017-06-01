; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define i1 @and_cmp_variable_power_of_two(i32 %x, i32 %y) {
  %shl = shl i32 1, %y
  %and = and i32 %x, %shl
  %cmp = icmp eq i32 %and, %shl
  ret i1 %cmp

; CHECK-LABEL: @and_cmp_variable_power_of_two
; CHECK: subfic 4, 4, 32
; CHECK: rlwnm 3, 3, 4, 31, 31
; CHECK: blr
}

define i1 @and_cmp_variable_power_of_two_64(i64 %x, i64 %y) {
  %shl = shl i64 1, %y
  %and = and i64 %x, %shl
  %cmp = icmp eq i64 %and, %shl
  ret i1 %cmp

; CHECK-LABEL: @and_cmp_variable_power_of_two_64
; CHECK: subfic 4, 4, 64
; CHECK: rldcl 3, 3, 4, 63
; CHECK: blr
}

define i1 @and_ncmp_variable_power_of_two(i32 %x, i32 %y) {
  %shl = shl i32 1, %y
  %and = and i32 %x, %shl
  %cmp = icmp ne i32 %and, %shl
  ret i1 %cmp

; CHECK-LABEL: @and_ncmp_variable_power_of_two
; CHECK-DAG: subfic 4, 4, 32
; CHECK-DAG: nor [[REG:[0-9]+]], 3, 3
; CHECK: rlwnm 3, [[REG]], 4, 31, 31
; CHECK: blr
}

define i1 @and_ncmp_variable_power_of_two_64(i64 %x, i64 %y) {
  %shl = shl i64 1, %y
  %and = and i64 %x, %shl
  %cmp = icmp ne i64 %and, %shl
  ret i1 %cmp

; CHECK-LABEL: @and_ncmp_variable_power_of_two_64
; CHECK-DAG: subfic 4, 4, 64
; CHECK-DAG: not [[REG:[0-9]+]], 3
; CHECK: rldcl 3, [[REG]], 4, 63
; CHECK: blr
}

