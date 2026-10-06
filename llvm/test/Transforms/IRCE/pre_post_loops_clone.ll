; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -S < %s 2>&1 | FileCheck %s
; This test is the same as pre_post_loops.ll, except that the loop body contains a token-generating
; call. We need to ensure that IRCE does not try to introduce a PHI or otherwise generate invalid IE.

declare token @llvm.source_token()
declare void @llvm.sink_token(token)

; CHECK: define void @test_01
define void @test_01(ptr %arr, ptr %a_len_ptr) {
entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  %token = call token @llvm.source_token()
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, 2147483647
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  call void @llvm.sink_token(token %token)
  ret void
}

define void @test_02(ptr %arr, ptr %a_len_ptr) {
entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 2147483647, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  %token = call token @llvm.source_token()
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp sgt i32 %idx.next, -1
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  call void @llvm.sink_token(token %token)
  ret void
}

!0 = !{i32 0, i32 50}
