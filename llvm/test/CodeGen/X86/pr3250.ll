; RUN: llvm-as < %s | llc -march=x86
; PR3250

declare i32 @safe_sub_func_short_u_u(i16 signext, i16 signext) nounwind

define i32 @func_106(i32 %p_107) nounwind {
entry:
        %0 = tail call i32 (...)* @safe_div_(i32 %p_107, i32 1) nounwind       
        ; <i32> [#uses=1]
        %1 = lshr i32 %0, -9            ; <i32> [#uses=1]
        %2 = trunc i32 %1 to i16                ; <i16> [#uses=1]
        %3 = tail call i32 @safe_sub_func_short_u_u(i16 signext 1, i16 signext
%2) nounwind             ; <i32> [#uses=0]
        ret i32 undef
}

declare i32 @safe_div_(...)
