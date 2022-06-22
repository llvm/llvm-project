; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse | FileCheck %s
; CHECK: movq ([[A0:%rdi|%rcx]]), %rax
; CHECK: movq 8([[A0]]), %rax
define i64 @foo_0(ptr %val) {
entry:
        %val12 = getelementptr <2 x i64>, ptr %val, i32 0, i32 0            ; <ptr> [#uses=1]
        %tmp7 = load i64, ptr %val12                ; <i64> [#uses=1]
        ret i64 %tmp7
}

define i64 @foo_1(ptr %val) {
entry:
        %tmp2.gep = getelementptr <2 x i64>, ptr %val, i32 0, i32 1         ; <ptr> [#uses=1]
        %tmp4 = load i64, ptr %tmp2.gep             ; <i64> [#uses=1]
        ret i64 %tmp4
}
