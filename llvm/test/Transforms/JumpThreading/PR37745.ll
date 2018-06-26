; RUN: opt -jump-threading -verify-each -S -mtriple=x86_64-- -o - %s

define void @foo() {
entry:
  br i1 false, label %A, label %B

A:
  %x = phi i32 [ undef, %entry ], [ %z, %B ]
  br label %B

B:
  %y = phi i32 [ undef, %entry ], [ %x, %A ]
  %z = add i32 %y, 1
  %cmp = icmp ne i32 %z, 0
  br i1 %cmp, label %exit, label %A

exit:
  ret void
}
