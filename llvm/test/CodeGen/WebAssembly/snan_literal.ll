; RUN: llc < %s --filetype=obj | llvm-objdump -d - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define float @float_sNaN() #0 {
entry:
; CHECK: 00 00 a0 7f
  ret float 0x7ff4000000000000
}

define float @float_qNaN() #0 {
entry:
; CHECK: 00 00 e0 7f
  ret float 0x7ffc000000000000
}


define double @double_sNaN() #0 {
entry:
; CHECK: 00 00 00 00 00 00 f4 7f
  ret double 0x7ff4000000000000
}

define double @double_qNaN() #0 {
entry:
; CHECK: 00 00 00 00 00 00 fc 7f
  ret double 0x7ffc000000000000
}

