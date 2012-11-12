; RUN: llc < %s -march=x86 -x86-asm-syntax=intel -mcpu=yonah | FileCheck %s

; Check that a fastcc function pops its stack variables before returning.

define x86_fastcallcc void @func(i64 inreg %X, i64 %Y, float %G, double %Z) nounwind {
        ret void
; CHECK: ret{{.*}}20
}

define x86_thiscallcc void @func2(i32 inreg %X, i64 %Y, float %G, double %Z) nounwind {
        ret void
; CHECK: ret{{.*}}20
}
