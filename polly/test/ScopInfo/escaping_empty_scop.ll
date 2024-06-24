; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void g();
;    int f(int *A) {
;      int a0 = 0, a1 = 0, a2 = 0;
;      for (int i = 0; i < 1000; i++) {
;        a0 = 2 * i;
;        // split
;        A[0] = i;
;        a1 = 2 * i;
;        // split
;        a2 = 2 * i;
;      }
;      g();
;      return a1 + a2;
;    }
;
; CHECK:      Stmt_bb1
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                { Stmt_bb1[i0] -> MemRef_a_0[] };
; CHECK:      Stmt_bb2
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                { Stmt_bb2[i0] -> MemRef_a_1[] };
; CHECK:      Stmt_bb3
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                { Stmt_bb3[i0] -> MemRef_a_2[] };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %i.0 = phi i32 [ 0, %bb ], [ %tmp4, %bb3 ]
  %a.0 = mul i32 %i.0, 2
  br label %bb2

bb2:                                              ; preds = %bb1
  %a.1 = mul i32 %i.0, 2
  store i32 %i.0, ptr %A, align 4
  br label %bb3

bb3:                                              ; preds = %bb2
  %tmp = shl nsw i32 %i.0, 1
  %tmp4 = add nuw nsw i32 %i.0, 1
  %a.2 = mul i32 %i.0, 2
  %exitcond = icmp ne i32 %i.0, 1000
  br i1 %exitcond, label %bb1, label %bb5

bb5:                                              ; preds = %bb1
  call void (...) @g() #2
  %add = add i32 %a.0, %a.1
  %add2 = add i32 %add, %a.2
  ret i32 %add2
}

declare void @g(...) #1
