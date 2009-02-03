; RUN: llvm-as < %s | \
; RUN:   llc -march=x86 -x86-asm-syntax=intel -mcpu=yonah | grep {ret	20}

; Check that a fastcc function pops its stack variables before returning.

define x86_fastcallcc void @func(i64 %X, i64 %Y, float %G, double %Z) nounwind {
        ret void
}
