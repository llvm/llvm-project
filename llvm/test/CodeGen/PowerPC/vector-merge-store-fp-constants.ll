; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: {{^}}merge_8_float_zero_stores:
; CHECK: li [[ZEROREG:[0-9]+]], 0
; CHECK-DAG: std [[ZEROREG]], 0([[PTR:[0-9]+]])
; CHECK-DAG: std [[ZEROREG]], 8([[PTR]])
; CHECK-DAG: std [[ZEROREG]], 16([[PTR]])
; CHECK-DAG: std [[ZEROREG]], 24([[PTR]])
; CHECK: blr
define void @merge_8_float_zero_stores(ptr %ptr) {
  %idx1 = getelementptr float, ptr %ptr, i64 1
  %idx2 = getelementptr float, ptr %ptr, i64 2
  %idx3 = getelementptr float, ptr %ptr, i64 3
  %idx4 = getelementptr float, ptr %ptr, i64 4
  %idx5 = getelementptr float, ptr %ptr, i64 5
  %idx6 = getelementptr float, ptr %ptr, i64 6
  %idx7 = getelementptr float, ptr %ptr, i64 7
  store float 0.0, ptr %ptr, align 4
  store float 0.0, ptr %idx1, align 4
  store float 0.0, ptr %idx2, align 4
  store float 0.0, ptr %idx3, align 4
  store float 0.0, ptr %idx4, align 4
  store float 0.0, ptr %idx5, align 4
  store float 0.0, ptr %idx6, align 4
  store float 0.0, ptr %idx7, align 4
  ret void
}
