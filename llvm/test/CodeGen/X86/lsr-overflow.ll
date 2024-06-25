; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s

; The comparison uses the pre-inc value, which could lead LSR to
; try to compute -INT64_MIN.

; CHECK: movabsq $-9223372036854775808, %rax
; CHECK: cmpq  %rax,
; CHECK: sete  %al

declare i64 @bar()

define i1 @foo() nounwind {
entry:
  br label %for.cond.i

for.cond.i:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.cond.i ]
  %t = call i64 @bar()
  %indvar.next = add i64 %indvar, 1
  %s = icmp ne i64 %indvar.next, %t
  br i1 %s, label %for.cond.i, label %__ABContainsLabel.exit

__ABContainsLabel.exit:
  %cmp = icmp eq i64 %indvar, 9223372036854775807
  ret i1 %cmp
}

define void @func_37() noreturn nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.inc8 ]
  %sub.i = add i64 undef, %indvar
  %cmp.i = icmp eq i64 %sub.i, -9223372036854775808
  br i1 undef, label %for.inc8, label %for.cond4

for.cond4:                                        ; preds = %for.cond4, %for.body
  br label %for.cond4

for.inc8:                                         ; preds = %for.body
  %indvar.next = add i64 %indvar, 1
  br label %for.body
}
