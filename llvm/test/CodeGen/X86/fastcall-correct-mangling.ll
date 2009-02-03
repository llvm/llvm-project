; RUN: llvm-as < %s | llc -march=x86 -mtriple=mingw32 | \
; RUN:   grep {@12}

; Check that a fastcall function gets correct mangling

define x86_fastcallcc void @func(i64 %X, i8 %Y, i8 %G, i16 %Z) {
        ret void
}

