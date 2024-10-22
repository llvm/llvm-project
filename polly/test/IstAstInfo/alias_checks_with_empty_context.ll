; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s \
; RUN:     | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = external local_unnamed_addr global i32, align 4
@global.1 = external local_unnamed_addr global i32, align 4

define void @hoge() local_unnamed_addr {
bb:
  %tmp = alloca i8, align 8
  br label %bb1

bb1:                                              ; preds = %bb19, %bb
  %tmp2 = phi i32 [ undef, %bb ], [ %tmp5, %bb19 ]
  %tmp3 = phi ptr [ @global, %bb ], [ %tmp20, %bb19 ]
  %tmp4 = icmp ugt i32 %tmp2, 5
  %tmp5 = select i1 %tmp4, i32 %tmp2, i32 5
  br label %bb6

bb6:                                              ; preds = %bb1
  br label %bb7

bb7:                                              ; preds = %bb10, %bb6
  %tmp8 = phi i8 [ 7, %bb6 ], [ %tmp11, %bb10 ]
  store i32 2, ptr %tmp3, align 4
  %tmp9 = load i8, ptr %tmp, align 8
  br label %bb10

bb10:                                             ; preds = %bb7
  store i32 undef, ptr @global.1, align 4
  %tmp11 = add nuw nsw i8 %tmp8, 1
  %tmp12 = icmp eq i8 %tmp11, 72
  br i1 %tmp12, label %bb13, label %bb7

bb13:                                             ; preds = %bb10
  %tmp14 = icmp eq i32 %tmp5, 0
  br i1 %tmp14, label %bb15, label %bb16

bb15:                                             ; preds = %bb13
  store i8 0, ptr %tmp, align 8
  br label %bb16

bb16:                                             ; preds = %bb15, %bb13
  br label %bb17

bb17:                                             ; preds = %bb16
  br i1 undef, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb19

bb19:                                             ; preds = %bb18, %bb17
  %tmp20 = phi ptr [ %tmp3, %bb17 ], [ @hoge, %bb18 ]
  br label %bb1
}

; CHECK: if (1 && (&MemRef_tmp3[1] <= &MemRef_global_1[0] || &MemRef_global_1[1] <= &MemRef_tmp3[0]) && (&MemRef_tmp[1] <= &MemRef_global_1[0] || &MemRef_global_1[1] <= &MemRef_tmp[0]) && (&MemRef_tmp[1] <= &MemRef_tmp3[0] || &MemRef_tmp3[1] <= &MemRef_tmp[0]))

; CHECK:            for (int c0 = 0; c0 <= 64; c0 += 1) {
; CHECK-NEXT:         Stmt_bb7(c0);
; CHECK-NEXT:         Stmt_bb10(c0);
; CHECK-NEXT:       }

; CHECK: else
; CHECK-NEXT:     {  /* original code */ }

