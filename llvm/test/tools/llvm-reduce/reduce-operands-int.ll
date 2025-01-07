; Test that llvm-reduce can reduce floating point operands
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ONE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; CHECK-INTERESTINGNESS: = add i32 %
; CHECK-INTERESTINGNESS: = add i32
; CHECK-INTERESTINGNESS: = add i32
; CHECK-INTERESTINGNESS: = add i32
; CHECK-INTERESTINGNESS: = add i32

; CHECK-INTERESTINGNESS: = add <2 x i32> %
; CHECK-INTERESTINGNESS: = add <2 x i32>
; CHECK-INTERESTINGNESS: = add <2 x i32>
; CHECK-INTERESTINGNESS: = add <2 x i32>
; CHECK-INTERESTINGNESS: = add <2 x i32>
; CHECK-INTERESTINGNESS: = add <2 x i32>

; CHECK-LABEL: define void @foo(


; ONE: %add0 = add i32 %arg0, 1
; ONE: %add1 = add i32 1, 1
; ONE: %add2 = add i32 1, 0
; ONE: %add3 = add i32 1, 1
; ONE: %add4 = add i32 1, 1
; ONE: %add5 = add <2 x i32> %arg2, splat (i32 1)
; ONE: %add6 = add <2 x i32> splat (i32 1), splat (i32 1)
; ONE: %add7 = add <2 x i32> splat (i32 1), zeroinitializer
; ONE: %add8 = add <2 x i32> splat (i32 1), splat (i32 1)
; ONE: %add9 = add <2 x i32> splat (i32 1), splat (i32 1)
; ONE: %add10 = add <2 x i32> splat (i32 1), splat (i32 1)


; ZERO: %add0 = add i32 %arg0, 0
; ZERO: %add1 = add i32 0, 0
; ZERO: %add2 = add i32 0, 0
; ZERO: %add3 = add i32 0, 0
; ZERO: %add4 = add i32 0, 0
; ZERO: %add5 = add <2 x i32> %arg2, zeroinitializer
; ZERO: %add6 = add <2 x i32> zeroinitializer, zeroinitializer
; ZERO: %add7 = add <2 x i32> zeroinitializer, zeroinitializer
; ZERO: %add8 = add <2 x i32> zeroinitializer, zeroinitializer
; ZERO: %add9 = add <2 x i32> zeroinitializer, zeroinitializer
; ZERO: %add10 = add <2 x i32> zeroinitializer, zeroinitializer

define void @foo(i32 %arg0, i32 %arg1, <2 x i32> %arg2, <2 x i32> %arg3) {
bb0:
  %add0 = add i32 %arg0, %arg1
  %add1 = add i32 %arg0, %arg1
  %add2 = add i32 %arg0, 0
  %add3 = add i32 %arg0, 1
  %add4 = add i32 %arg0, undef
  %add5 = add <2 x i32> %arg2, %arg3
  %add6 = add <2 x i32> %arg2, %arg3
  %add7 = add <2 x i32> %arg2, zeroinitializer
  %add8 = add <2 x i32> %arg2, <i32 1, i32 1>
  %add9 = add <2 x i32> %arg2, undef
  %add10 = add <2 x i32> %arg2, <i32 4, i32 6>
  ret void
}
