; Test the passing of scalar i128 values on z/OS.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck %s

; CHECK-LABEL: call_i128 DS 0H
; CHECK-DAG: larl    1,L#CPI0_0
; CHECK-DAG: vl      24,0(1),3
; CHECK-DAG: larl    1,L#CPI0_1
; CHECK-DAG: vl      25,0(1),3

define i128 @call_i128() {
entry:
  %retval = call i128 (i128, i128) @pass_i128(i128 64, i128 65)
  ret i128 %retval
}

; CHECK-LABEL: pass_i128 DS 0H
; CHECK: vaq     24,24,25
define i128 @pass_i128(i128 %arg0, i128 %arg1) {
entry:
  %N = add i128 %arg0, %arg1
  ret i128 %N
}
