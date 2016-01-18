; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Check that the booleans are converted using zext and not via sext.
; 0x1 means that we only look at the first bit.

;CHECK: 0x1
;CHECK-LABEL: ui_to_fp_conv:
;CHECK: ret
define void @ui_to_fp_conv(<8 x float> * nocapture %aFOO, <8 x float>* nocapture %RET) nounwind {
allocas:
  %bincmp = fcmp olt <8 x float> <float 1.000000e+00, float 1.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00> , <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %bool2float = uitofp <8 x i1> %bincmp to <8 x float>
  store <8 x float> %bool2float, <8 x float>* %RET, align 4
  ret void
}



