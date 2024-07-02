; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>,scop(print<polly-simplify>)' -disable-output < %s 2>&1 | FileCheck %s
;
; %tmp5 must keep the Value WRITE MemoryAccess, because as an incoming value of
; %tmp4, it is an "external use".
;
; A common mistake is to assume that %tmp5 is used by %tmp4 in bb3, when
; practically it's the incoming block %bb9 which is the user.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge() {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp = load ptr, ptr undef
  br label %bb3

bb3:                                              ; preds = %bb9, %bb2
  %tmp4 = phi ptr [ %tmp, %bb2 ], [ %tmp5, %bb9 ]
  %tmp5 = getelementptr inbounds i64, ptr %tmp4, i64 1
  %tmp6 = load i64, ptr %tmp5
  %tmp7 = and i64 %tmp6, 4160749568
  br i1 false, label %bb8, label %bb9

bb8:                                              ; preds = %bb3
  br label %bb9

bb9:                                              ; preds = %bb8, %bb3
  %tmp10 = icmp eq i64 %tmp7, 134217728
  br i1 %tmp10, label %bb11, label %bb3

bb11:                                             ; preds = %bb9
  br label %bb12

bb12:                                             ; preds = %bb11
  ret void
}


; CHECK:      MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:     [p_0] -> { Stmt_bb3[] -> MemRef_tmp5[] };

; CHECK: SCoP could not be simplified
