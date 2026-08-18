; RUN: opt -passes=metarenamer -S < %s | FileCheck %s

define void @opcodes(ptr %p, ptr %arr) {
; CHECK-LABEL: bbl:
; CHECK:         %load = load i32, ptr %arg, align 4
; CHECK:         br label %bbl2
; CHECK-LABEL: bbl2:                                              ; preds = %bbl5, %bbl
; CHECK:         %phi = phi i32 [ %load, %bbl ], [ %sub, %bbl5 ]
; CHECK:         %icmp = icmp eq i32 %phi, 0
; CHECK:         br i1 %icmp, label %bbl8, label %bbl3
; CHECK-LABEL: bbl3:                                              ; preds = %bbl2
; CHECK:         %sub = sub i32 %phi, 1
; CHECK:         %icmp4 = icmp ult i32 %sub, %load
; CHECK:         br i1 %icmp4, label %bbl5, label %bbl9
; CHECK-LABEL: bbl5:                                              ; preds = %bbl3
; CHECK:         %getelementptr = getelementptr i32, ptr %arg, i32 %phi
; CHECK:         %load6 = load i32, ptr %getelementptr, align 4
; CHECK:         %icmp7 = icmp eq i32 %load6, 0
; CHECK:         br i1 %icmp7, label %bbl2, label %bbl8
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
