; RUN: llc -o - -verify-machineinstrs %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@data = external dso_local global [3 x i32], align 4
@store = external dso_local global i32, align 4

; %else1 and %then2 end up lowering to identical blocks. These blocks should be
; merged during tail-merging.
; CHECK-LABEL: merge_identical_blocks
; CHECK: movl $data+4
; CHECK-NOT: movl $data+4
; CHECK: retq
define void @merge_identical_blocks(i1 %a, i1 %b) {
entry:
  br label %if1

if1:                                              ; predfs = %entry
  br i1 %a, label %else1, label %if2

else1:                                            ; preds = %if1
  %ptr.else1 = getelementptr inbounds [3 x i32], [3 x i32]* @data, i64 0, i32 1
  br label %phi_join

if2:                                              ; preds = %if1
  br i1 %b, label %then2, label %else2

then2:                                            ; preds = %if2
  %ptr.then2 = getelementptr inbounds [3 x i32], [3 x i32]* @data, i64 0, i32 1
  br label %phi_join

else2:                                            ; preds = %if2
  %ptr.else2 = getelementptr inbounds [3 x i32], [3 x i32]* @data, i64 0, i32 2
  br label %phi_join

phi_join:                                         ; preds = %else1, %then2, %else2
  %val.ptr = phi i32* [ %ptr.else1, %else1 ], [ %ptr.then2, %then2 ], [ %ptr.else2, %else2 ]
  %val = load i32, i32* %val.ptr, align 4
  store i32 %val, i32* @store, align 4
  ret void
}
