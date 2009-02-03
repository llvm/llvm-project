; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep {add	ESP, 8}

target triple = "i686-pc-linux-gnu"

declare x86_fastcallcc void @func(i32*, i64)

define x86_fastcallcc void @caller(i32, i64) {
        %X = alloca i32         ; <i32*> [#uses=1]
        call x86_fastcallcc void @func( i32* %X, i64 0 )
        ret void
}

