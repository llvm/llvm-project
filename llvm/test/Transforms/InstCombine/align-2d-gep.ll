; RUN: opt < %s -instcombine -S | grep "align 16" | count 1
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; A multi-dimensional array in a nested loop doing vector stores that
; aren't yet aligned. Instcombine can understand the addressing in the
; Nice case to prove 16 byte alignment. In the Awkward case, the inner
; array dimension is not even, so the stores to it won't always be
; aligned. Instcombine should prove alignment in exactly one of the two
; stores.

@Nice    = global [1001 x [20000 x double]] zeroinitializer, align 32
@Awkward = global [1001 x [20001 x double]] zeroinitializer, align 32

define void @foo() nounwind  {
entry:
  br label %bb7.outer

bb7.outer:
  %i = phi i64 [ 0, %entry ], [ %indvar.next26, %bb11 ]
  br label %bb1

bb1:
  %j = phi i64 [ 0, %bb7.outer ], [ %indvar.next, %bb1 ]

  %t4 = getelementptr [1001 x [20000 x double]]* @Nice, i64 0, i64 %i, i64 %j
  %q = bitcast double* %t4 to <2 x double>*
  store <2 x double><double 0.0, double 0.0>, <2 x double>* %q, align 8

  %s4 = getelementptr [1001 x [20001 x double]]* @Awkward, i64 0, i64 %i, i64 %j
  %r = bitcast double* %s4 to <2 x double>*
  store <2 x double><double 0.0, double 0.0>, <2 x double>* %r, align 8

  %indvar.next = add i64 %j, 2
  %exitcond = icmp eq i64 %indvar.next, 557
  br i1 %exitcond, label %bb11, label %bb1

bb11:
  %indvar.next26 = add i64 %i, 1
  %exitcond27 = icmp eq i64 %indvar.next26, 991
  br i1 %exitcond27, label %return.split, label %bb7.outer

return.split:
  ret void
}
