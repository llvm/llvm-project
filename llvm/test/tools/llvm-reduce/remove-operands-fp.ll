; Test that llvm-reduce can reduce floating point operands
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ONE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; CHECK-INTERESTINGNESS: = fadd float %
; CHECK-INTERESTINGNESS: = fadd float
; CHECK-INTERESTINGNESS: = fadd float
; CHECK-INTERESTINGNESS: = fadd float
; CHECK-INTERESTINGNESS: = fadd float
; CHECK-INTERESTINGNESS: = fadd float

; CHECK-INTERESTINGNESS: = fadd <2 x float> %
; CHECK-INTERESTINGNESS: = fadd <2 x float>
; CHECK-INTERESTINGNESS: = fadd <2 x float>
; CHECK-INTERESTINGNESS: = fadd <2 x float>
; CHECK-INTERESTINGNESS: = fadd <2 x float>
; CHECK-INTERESTINGNESS: = fadd <2 x float>

; CHECK-LABEL: define void @foo(


; ONE: %fadd0 = fadd float %arg0, 1.000000e+00
; ONE: %fadd1 = fadd float 1.000000e+00, 1.000000e+00
; ONE: %fadd2 = fadd float 1.000000e+00, 0.000000e+00
; ONE: %fadd3 = fadd float 1.000000e+00, 1.000000e+00
; ONE: %fadd4 = fadd float 1.000000e+00, 1.000000e+00
; ONE: %fadd5 = fadd float 1.000000e+00, 1.000000e+00
; ONE: %fadd6 = fadd <2 x float> %arg2, <float 1.000000e+00, float 1.000000e+00>
; ONE: %fadd7 = fadd <2 x float> <float 1.000000e+00, float 1.000000e+00>, <float 1.000000e+00, float 1.000000e+00>
; ONE: %fadd8 = fadd <2 x float> <float 1.000000e+00, float 1.000000e+00>, zeroinitializer
; ONE: %fadd9 = fadd <2 x float> <float 1.000000e+00, float 1.000000e+00>, <float 1.000000e+00, float 1.000000e+00>
; ONE: %fadd10 = fadd <2 x float> <float 1.000000e+00, float 1.000000e+00>, <float 1.000000e+00, float 1.000000e+00>
; ONE: %fadd11 = fadd <2 x float> <float 1.000000e+00, float 1.000000e+00>, <float 1.000000e+00, float 1.000000e+00>


; ZERO: %fadd0 = fadd float %arg0, 0.000000e+00
; ZERO: %fadd1 = fadd float 0.000000e+00, 0.000000e+00
; ZERO: %fadd2 = fadd float 0.000000e+00, 0.000000e+00
; ZERO: %fadd3 = fadd float 0.000000e+00, 0.000000e+00
; ZERO: %fadd4 = fadd float 0.000000e+00, 0.000000e+00
; ZERO: %fadd5 = fadd float 0.000000e+00, 0.000000e+00
; ZERO: %fadd6 = fadd <2 x float> %arg2, zeroinitializer
; ZERO: %fadd7 = fadd <2 x float> zeroinitializer, zeroinitializer
; ZERO: %fadd8 = fadd <2 x float> zeroinitializer, zeroinitializer
; ZERO: %fadd9 = fadd <2 x float> zeroinitializer, zeroinitializer
; ZERO: %fadd10 = fadd <2 x float> zeroinitializer, zeroinitializer
; ZERO: %fadd11 = fadd <2 x float> zeroinitializer, zeroinitializer

define void @foo(float %arg0, float %arg1, <2 x float> %arg2, <2 x float> %arg3) {
bb0:
  %fadd0 = fadd float %arg0, %arg1
  %fadd1 = fadd float %arg0, %arg1
  %fadd2 = fadd float %arg0, 0.0
  %fadd3 = fadd float %arg0, 1.0
  %fadd4 = fadd float %arg0, 0x7FF8000000000000
  %fadd5 = fadd float %arg0, undef
  %fadd6 = fadd <2 x float> %arg2, %arg3
  %fadd7 = fadd <2 x float> %arg2, %arg3
  %fadd8 = fadd <2 x float> %arg2, zeroinitializer
  %fadd9 = fadd <2 x float> %arg2, <float 1.0, float 1.0>
  %fadd10 = fadd <2 x float> %arg2, undef
  %fadd11 = fadd <2 x float> %arg2, <float 0x7FF8000000000000, float 0x7FF8000000000000>
  ret void
}
