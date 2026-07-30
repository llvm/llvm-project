; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: global double -0.000000e+00
@0 = global double 0x8000000000000000

; CHECK: global float -0.000000e+00
@1 = global float -0.0

