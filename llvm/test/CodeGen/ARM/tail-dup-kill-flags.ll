; RUN: llc %s -o - -fast-isel=true -O1 -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios8.0.0"

; Tail duplication was incorrectly leaving kill flags on the duplicated instructions.
; The machine verifier is able to spot this error, so this test should pass if it passes verification.

; CHECK-LABEL: @test

%struct.cdiff_ctx = type { i8*, %struct.cdiff_node*, %struct.cdiff_node*, %struct.cdiff_node*, %struct.cdiff_node* }
%struct.cdiff_node = type { i32, i8*, i8*, %struct.cdiff_node* }

declare i32 @logg(i32)

define hidden i32 @test(%struct.cdiff_ctx* nocapture %ctx, %struct.cdiff_node* %tmp10) {
bb:
  br label %.outer

bb33:                                             ; preds = %bb92, %.outer
  %lines.0 = phi i32 [ %tmp37, %bb92 ], [ %lines.0.ph, %.outer ]
  %tmp37 = add i32 %lines.0, 1
  %tmp39 = load i32, i32* %tmp57, align 4
  %tmp40 = icmp eq i32 %tmp39, %tmp37
  br i1 %tmp40, label %bb41, label %bb92

bb41:                                             ; preds = %bb33
  %tmp45 = call i32 @strncmp()
  %tmp46 = icmp eq i32 %tmp45, 0
  br i1 %tmp46, label %bb53, label %bb47

bb47:                                             ; preds = %bb41
  %tmp52 = call i32 @logg(i32 %tmp37)
  ret i32 -1

bb53:                                             ; preds = %bb41
  %tmp54 = getelementptr inbounds %struct.cdiff_node, %struct.cdiff_node* %del.0.ph, i32 0, i32 3
  %tmp55 = load %struct.cdiff_node*, %struct.cdiff_node** %tmp54, align 4
  br label %.outer

.outer:                                           ; preds = %bb53, %bb
  %del.0.ph = phi %struct.cdiff_node* [ %tmp55, %bb53 ], [ null, %bb ]
  %lines.0.ph = phi i32 [ 1, %bb53 ], [ 0, %bb ]
  %tmp57 = getelementptr inbounds %struct.cdiff_node, %struct.cdiff_node* %del.0.ph, i32 0, i32 0
  br label %bb33

bb92:                                             ; preds = %bb33
  %tmp93 = call i32 @puts()
  br label %bb33
}

declare i32 @strncmp()

declare i32 @puts()
