; This shouldn't crash
; RUN: llvm-as < %s | llc -march=alpha 

target datalayout = "e-p:64:64"
target triple = "alphaev6-unknown-linux-gnu"
deplibs = [ "c", "crtend", "stdc++" ]
        %struct.__va_list_tag = type { i8*, i32 }

define i32 @emit_library_call_value(i32 %nargs, ...) {
entry:
        %tmp.223 = va_arg %struct.__va_list_tag* null, i32              ; <i32> [#uses=1]
        ret i32 %tmp.223
}

