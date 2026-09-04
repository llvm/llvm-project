; RUN: opt -passes=licm -S < %s | FileCheck %s

define void @query_stays_in_loop(i32 %count) {
; CHECK-LABEL: define void @query_stays_in_loop(
; CHECK:       loop:
; CHECK:         call i1 @llvm.is.debugging.enabled()
; CHECK:         br i1 {{.*}}, label %loop, label %exit
; CHECK:       exit:
;
entry:
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit

loop:
  %index = phi i32 [ 0, %entry ], [ %next, %loop ]
  %enabled = call i1 @llvm.is.debugging.enabled()
  %next = add nuw i32 %index, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

declare i1 @llvm.is.debugging.enabled()
