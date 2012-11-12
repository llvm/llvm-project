; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | FileCheck %s
; check that fastcc is passing stuff in regs.

declare x86_fastcallcc i64 @callee(i64 inreg)

define i64 @caller() {
        %X = call x86_fastcallcc  i64 @callee( i64 4294967299 )          ; <i64> [#uses=1]
; CHECK: mov{{.*}}EDX, 1
        ret i64 %X
}

define x86_fastcallcc i64 @caller2(i64 inreg %X) {
        ret i64 %X
; CHECK: mov{{.*}}EAX, ECX
}

declare x86_thiscallcc i64 @callee2(i32)

define i64 @caller3() {
        %X = call x86_thiscallcc i64 @callee2( i32 3 )
; CHECK: mov{{.*}}ECX, 3
        ret i64 %X
}

define x86_thiscallcc i32 @caller4(i32 %X) {
        ret i32 %X
; CHECK: mov{{.*}}EAX, ECX
}

