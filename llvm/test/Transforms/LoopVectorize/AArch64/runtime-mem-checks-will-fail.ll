; RUN: opt -passes=loop-vectorize -S -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s

target triple = "arm64-apple-macosx"

; CHECK: remark: <unknown>:0:0: loop not vectorized: runtime checks are known to fail, so we will never enter the vector loop
define void @foo(ptr %a, i64 %off, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.off = add i64 %iv, %off
  %p.off = getelementptr i32, ptr %a, i64 %iv.off
  %v0 = load i32, ptr %p.off
  %p0 = getelementptr i32, ptr %a, i64 %iv
  %v1 = load i16, ptr %p0
  %v1.ext = sext i16 %v1 to i32
  %s = add i32 %v0, %v1.ext
  %iv.next = add i64 %iv, 1
  %p1 = getelementptr i32, ptr %a, i64 %iv.next
  store i32 %s, ptr %p1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
