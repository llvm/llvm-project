; RUN: opt -passes=metarenamer -S < %s | FileCheck %s

define void @opcodes(ptr %p, ptr %arr) {
; CHECK-LABEL: bb:
; CHECK:         %load = load i32, ptr %arg, align 4
; CHECK:         br label %bb2
; CHECK-LABEL: bb2:                                              ; preds = %bb5, %bb
; CHECK:         %phi = phi i32 [ %load, %bb ], [ %sub, %bb5 ]
; CHECK:         %icmp = icmp eq i32 %phi, 0
; CHECK:         br i1 %icmp, label %bb8, label %bb3
; CHECK-LABEL: bb3:                                              ; preds = %bb2
; CHECK:         %sub = sub i32 %phi, 1
; CHECK:         %icmp4 = icmp ult i32 %sub, %load
; CHECK:         br i1 %icmp4, label %bb5, label %bb9
; CHECK-LABEL: bb5:                                              ; preds = %bb3
; CHECK:         %getelementptr = getelementptr i32, ptr %arg, i32 %phi
; CHECK:         %load6 = load i32, ptr %getelementptr, align 4
; CHECK:         %icmp7 = icmp eq i32 %load6, 0
; CHECK:         br i1 %icmp7, label %bb2, label %bb8
preheader:
  %len = load i32, ptr %p
  br label %loop

loop:
  %iv = phi i32 [%len, %preheader], [%iv.next, %backedge]
  %zero_cond = icmp eq i32 %iv, 0
  br i1 %zero_cond, label %exit, label %range_check_block

range_check_block:
  %iv.next = sub i32 %iv, 1
  %range_check = icmp ult i32 %iv.next, %len
  br i1 %range_check, label %backedge, label %fail

backedge:
  %el.ptr = getelementptr i32, ptr %p, i32 %iv
  %el = load i32, ptr %el.ptr
  %loop.cond = icmp eq i32 %el, 0
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void

fail:
  unreachable
}
