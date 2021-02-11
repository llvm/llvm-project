; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o - -global-isel < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

define i32 @test_01(i32* %p, i64 %len, i32 %x) {
; CHECK-LABEL: test_01

entry:
  %scevgep = getelementptr i32, i32* %p, i64 -1
  br label %loop

loop:                                             ; preds = %backedge, %entry
  %iv = phi i64 [ %iv.next, %backedge ], [ %len, %entry ]
  %iv.next = add i64 %iv, -1
  %cond_1 = icmp eq i64 %iv, 0
  br i1 %cond_1, label %exit, label %backedge

backedge:                                         ; preds = %loop
  %scevgep1 = getelementptr i32, i32* %scevgep, i64 %iv
  %loaded = load atomic i32, i32* %scevgep1 unordered, align 4
  %cond_2 = icmp eq i32 %loaded, %x
  br i1 %cond_2, label %failure, label %loop

exit:                                             ; preds = %loop
  ret i32 -1

failure:
  unreachable
}

