; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; CHECK-NOT: {{addq.*8}}
; CHECK:     ({{%rdi|%rcx}},%rax,8)
; CHECK-NOT: {{addq.*8}}

define void @foo(double* %y) nounwind {
entry:
        br label %bb

bb:
        %i = phi i64 [ 0, %entry ], [ %k, %bb ]
        %j = getelementptr double* %y, i64 %i
        store double 0.000000e+00, double* %j
        %k = add i64 %i, 1
        %n = icmp eq i64 %k, 0
        br i1 %n, label %return, label %bb

return:
        ret void
}

