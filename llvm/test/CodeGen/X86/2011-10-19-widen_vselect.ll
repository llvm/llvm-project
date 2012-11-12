; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Make sure that we don't crash when legalizng vselect and vsetcc and that
; we are able to generate vector blend instructions.

; CHECK: simple_widen
; CHECK: blend
; CHECK: ret
define void @simple_widen() {
entry:
  %0 = select <2 x i1> undef, <2 x float> undef, <2 x float> undef
  store <2 x float> %0, <2 x float>* undef
  ret void
}

; CHECK: complex_inreg_work
; CHECK: blend
; CHECK: ret

define void @complex_inreg_work() {
entry:
  %0 = fcmp oeq <2 x float> undef, undef
  %1 = select <2 x i1> %0, <2 x float> undef, <2 x float> undef
  store <2 x float> %1, <2 x float>* undef
  ret void
}

; CHECK: zero_test
; CHECK: blend
; CHECK: ret

define void @zero_test() {
entry:
  %0 = select <2 x i1> undef, <2 x float> undef, <2 x float> zeroinitializer
  store <2 x float> %0, <2 x float>* undef
  ret void
}

; CHECK: full_test
; CHECK: blend
; CHECK: ret

define void @full_test() {
 entry:
   %Cy300 = alloca <4 x float>
   %Cy11a = alloca <2 x float>
   %Cy118 = alloca <2 x float>
   %Cy119 = alloca <2 x float>
   br label %B1

 B1:                                               ; preds = %entry
   %0 = load <2 x float>* %Cy119
   %1 = fptosi <2 x float> %0 to <2 x i32>
   %2 = sitofp <2 x i32> %1 to <2 x float>
   %3 = fcmp ogt <2 x float> %0, zeroinitializer
   %4 = fadd <2 x float> %2, <float 1.000000e+00, float 1.000000e+00>
   %5 = select <2 x i1> %3, <2 x float> %4, <2 x float> %2
   %6 = fcmp oeq <2 x float> %2, %0
   %7 = select <2 x i1> %6, <2 x float> %0, <2 x float> %5
   store <2 x float> %7, <2 x float>* %Cy118
   %8 = load <2 x float>* %Cy118
   store <2 x float> %8, <2 x float>* %Cy11a
   ret void
}


