; RUN: llc -verify-machineinstrs -mtriple=ppc64-apple-darwin < %s | FileCheck %s

; CHECK-LABEL: {{^}}_merge_8_float_zero_stores:
; CHECK: li [[ZEROREG:r[0-9]+]], 0
; CHECK-DAG: std [[ZEROREG]], 0([[PTR:r[0-9]+]])
; CHECK-DAG: std [[ZEROREG]], 8([[PTR]])
; CHECK-DAG: std [[ZEROREG]], 16([[PTR]])
; CHECK-DAG: std [[ZEROREG]], 24([[PTR]])
; CHECK: blr
define void @merge_8_float_zero_stores(float* %ptr) {
  %idx0 = getelementptr float, float* %ptr, i64 0
  %idx1 = getelementptr float, float* %ptr, i64 1
  %idx2 = getelementptr float, float* %ptr, i64 2
  %idx3 = getelementptr float, float* %ptr, i64 3
  %idx4 = getelementptr float, float* %ptr, i64 4
  %idx5 = getelementptr float, float* %ptr, i64 5
  %idx6 = getelementptr float, float* %ptr, i64 6
  %idx7 = getelementptr float, float* %ptr, i64 7
  store float 0.0, float* %idx0, align 4
  store float 0.0, float* %idx1, align 4
  store float 0.0, float* %idx2, align 4
  store float 0.0, float* %idx3, align 4
  store float 0.0, float* %idx4, align 4
  store float 0.0, float* %idx5, align 4
  store float 0.0, float* %idx6, align 4
  store float 0.0, float* %idx7, align 4
  ret void
}
