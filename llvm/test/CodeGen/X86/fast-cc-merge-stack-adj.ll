; RUN: llc < %s -mcpu=generic -x86-asm-syntax=intel | FileCheck %s
; CHECK: add esp, 8

target triple = "i686-pc-linux-gnu"

declare x86_fastcallcc void @func(ptr, i64 inreg)

define x86_fastcallcc void @caller(i32, i64) {
        %X = alloca i32         ; <ptr> [#uses=1]
        call x86_fastcallcc void @func( ptr %X, i64 inreg 0 )
        ret void
}

