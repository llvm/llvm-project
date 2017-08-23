; RUN: llc -verify-machineinstrs < %s -march=ppc64le | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -march=ppc64 | FileCheck %s

define i64 @ori_test_a(i64 %a) {
entry:
; CHECK-LABEL: @ori_test_a
; CHECK-DAG:  ori 3, 3, 65535
; CHECK-DAG:  oris 3, 3, 65535
; CHECK-NEXT:  blr
  %or = or i64 %a, 4294967295
  ret i64 %or
}

define i64 @ori_test_b(i64 %a) {
entry:
; CHECK-LABEL: @ori_test_b
; CHECK:  or 3, 3, {{[0-9]+}}
; CHECK-NEXT:  blr
  %or = or i64 %a, 4294967296
  ret i64 %or
}

define i64 @ori_test_c(i64 %a) {
entry:
; CHECK-LABEL: @ori_test_c
; CHECK:  ori 3, 3, 65535
; CHECK-NEXT:  blr
  %or = or i64 %a, 65535
  ret i64 %or
}

define i64 @ori_test_d(i64 %a) {
entry:
; CHECK-LABEL: @ori_test_d
; CHECK:  oris 3, 3, 1
; CHECK-NEXT:  blr
  %or = or i64 %a, 65536
  ret i64 %or
}

define zeroext i32 @ori_test_e(i32 zeroext %a) {
entry:
; CHECK-LABEL: @ori_test_e
; CHECK-DAG:  ori 3, 3, 65535
; CHECK-DAG:  oris 3, 3, 255
; CHECK-NEXT:  blr
  %or = or i32 %a, 16777215
  ret i32 %or
}

define i64 @xori_test_a(i64 %a) {
entry:
; CHECK-LABEL: @xori_test_a
; CHECK-DAG:  xori 3, 3, 65535
; CHECK-DAG:  xoris 3, 3, 65535
; CHECK-NEXT:  blr
  %xor = xor i64 %a, 4294967295
  ret i64 %xor
}

define i64 @xori_test_b(i64 %a) {
entry:
; CHECK-LABEL: @xori_test_b
; CHECK:  xor 3, 3, {{[0-9]+}}
; CHECK-NEXT:  blr
  %xor = xor i64 %a, 4294967296
  ret i64 %xor
}

define i64 @xori_test_c(i64 %a) {
entry:
; CHECK-LABEL: @xori_test_c
; CHECK:  xori 3, 3, 65535
; CHECK-NEXT:  blr
  %xor = xor i64 %a, 65535
  ret i64 %xor
}

define i64 @xori_test_d(i64 %a) {
entry:
; CHECK-LABEL: @xori_test_d
; CHECK:  xoris 3, 3, 1
; CHECK-NEXT:  blr
  %xor = xor i64 %a, 65536
  ret i64 %xor
}

define zeroext i32 @xori_test_e(i32 zeroext %a) {
entry:
; CHECK-LABEL: @xori_test_e
; CHECK-DAG:  xori 3, 3, 65535
; CHECK-DAG:  xoris 3, 3, 255
; CHECK-NEXT:  blr
  %xor = xor i32 %a, 16777215
  ret i32 %xor
}
