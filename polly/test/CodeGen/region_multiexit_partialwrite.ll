; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; This text case has a partial write of PHI in a region-statement. It
; requires that the new PHINode from the region's exiting block is
; generated before before the partial memory write.
;
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @region_multiexit_partialwrite(i32* %arg, i64 %arg1, i32* %arg2) {
bb:
  br label %bb3

bb3:
  %tmp = phi i64 [ %tmp17, %bb10 ], [ 1, %bb ]
  %tmp4 = getelementptr inbounds i32, i32* %arg, i64 %tmp
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp6 = icmp slt i32 %tmp5, 0
  br i1 %tmp6, label %bb7, label %bb9

bb7:
  %tmp8 = select i1 undef, i32 -2147483648, i32 undef
  br label %bb10

bb9:
  br label %bb10

bb10:
  %tmp11 = phi i32 [ %tmp8, %bb7 ], [ undef, %bb9 ]
  %tmp16 = getelementptr inbounds i32, i32* %arg2, i64 %tmp
  store i32 %tmp11, i32* %tmp16, align 4
  %tmp17 = add nuw i64 %tmp, 1
  %tmp18 = icmp eq i64 %tmp17, %arg1
  br i1 %tmp18, label %bb19, label %bb3

bb19:
  ret void
}


; CHECK:      polly.stmt.bb10.exit:
; CHECK-NEXT:   %polly.tmp11 = phi i32 [ %p_tmp8, %polly.stmt.bb7 ], [ undef, %polly.stmt.bb9 ]

; CHECK: polly.stmt.bb10.exit.Stmt_bb3__TO__bb10_Write1.partial:
; CHECK:   store i32 %polly.tmp11
