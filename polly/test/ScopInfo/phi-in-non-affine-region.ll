; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

; Verify that 'tmp' is stored in bb1 and read by bb3, as it is needed as
; incoming value for the tmp11 PHI node.

; CHECK: Stmt_bb1
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_bb1[] };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_bb1[] -> [0] };
; CHECK-NEXT:   ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_bb1[] -> MemRef_global[0] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:       { Stmt_bb1[] -> MemRef_tmp[] };
; CHECK-NEXT: Stmt_bb3__TO__bb10
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_bb3__TO__bb10[] };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_bb3__TO__bb10[] -> [1] };
; CHECK-NEXT:   ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:       { Stmt_bb3__TO__bb10[] -> MemRef_tmp[] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:       { Stmt_bb3__TO__bb10[] -> MemRef_tmp11[] };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { double, double, i8, i8, i8 }

@global = external local_unnamed_addr global %struct.hoge, align 8

define void @widget() local_unnamed_addr {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = load double, ptr @global, align 8
  br i1 false, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  br i1 false, label %bb8, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb4
  %tmp6 = and i32 undef, 16711680
  %tmp7 = icmp eq i32 %tmp6, 0
  br i1 %tmp7, label %bb8, label %bb10

bb8:                                              ; preds = %bb5, %bb3
  %tmp9 = phi double [ %tmp, %bb3 ], [ undef, %bb5 ]
  br label %bb10

bb10:                                             ; preds = %bb8, %bb5
  %tmp11 = phi double [ undef, %bb5 ], [ %tmp9, %bb8 ]
  ret void
}
