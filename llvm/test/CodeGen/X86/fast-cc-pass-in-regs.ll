; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep {mov	EDX, 1}
; check that fastcc is passing stuff in regs.

declare x86_fastcallcc i64 @callee(i64)

define i64 @caller() {
        %X = call x86_fastcallcc  i64 @callee( i64 4294967299 )          ; <i64> [#uses=1]
        ret i64 %X
}

define x86_fastcallcc i64 @caller2(i64 %X) {
        ret i64 %X
}

