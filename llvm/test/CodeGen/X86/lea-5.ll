; test for more complicated forms of lea operands which can be generated
; in loop optimized cases.
; See also http://llvm.org/bugs/show_bug.cgi?id=20016

; RUN: llc < %s -mtriple=x86_64-linux -O2        | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -O2 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-nacl -O2 | FileCheck %s -check-prefix=X32

; Function Attrs: nounwind readnone uwtable
define void @foo(i32 %x, i32 %d) #0 {
entry:
  %a = alloca [8 x i32], align 16
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %d.addr.0 = phi i32 [ %d, %entry ], [ %inc, %while.cond ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %d.addr.0

; CHECK: leaq	-40(%rsp,%r{{[^,]*}},4), %rax
; X32:   leal	-40(%rsp,%r{{[^,]*}},4), %eax
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp eq i32 %0, 0
  %inc = add nsw i32 %d.addr.0, 1

; CHECK: leaq	4(%r{{[^,]*}}), %r{{[^,]*}}
; X32:   leal	4(%r{{[^,]*}}), %e{{[^,]*}}
  br i1 %cmp1, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

; The same test as above but with enforsed stack realignment (%a aligned by 64)
; to check one more case of correct lea generation.

; Function Attrs: nounwind readnone uwtable
define void @bar(i32 %x, i32 %d) #0 {
entry:
  %a = alloca [8 x i32], align 64
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %d.addr.0 = phi i32 [ %d, %entry ], [ %inc, %while.cond ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %d.addr.0

; CHECK: leaq	(%rsp,%r{{[^,]*}},4), %rax
; X32:   leal	(%rsp,%r{{[^,]*}},4), %eax
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp eq i32 %0, 0
  %inc = add nsw i32 %d.addr.0, 1

; CHECK: leaq	4(%r{{[^,]*}}), %r{{[^,]*}}
; X32:   leal	4(%r{{[^,]*}}), %e{{[^,]*}}
  br i1 %cmp1, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

