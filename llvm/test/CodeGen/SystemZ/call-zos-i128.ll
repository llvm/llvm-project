; Test the passing of scalar i128 values on z/OS.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck %s

; CHECK-LABEL: call_i128:
; CHECK-DAG: larl    1, @CPI0_0
; CHECK-DAG: vl      0, 0(1), 3
; CHECK-DAG: vst     0, 2256(4), 3
; CHECK-DAG: larl    1, @CPI0_1
; CHECK-DAG: vl      0, 0(1), 3
; CHECK-DAG: vst     0, 2272(4), 3
; CHECK-DAG: la      1, 2288(4)
; CHECK-DAG: la      2, 2272(4)
; CHECK-DAG: la      3, 2256(4)

define i128 @call_i128() {
entry:
  %retval = call i128 (i128, i128) @pass_i128(i128 64, i128 65)
  ret i128 %retval
}

; CHECK-LABEL: pass_i128:
; CHECK: vl      0, 0(3), 3
; CHECK: vl      1, 0(2), 3
; CHECK: vaq     0, 1, 0
; CHECK: vst     0, 0(1), 3
define i128 @pass_i128(i128 %arg0, i128 %arg1) {
entry:
  %N = add i128 %arg0, %arg1
  ret i128 %N
}
