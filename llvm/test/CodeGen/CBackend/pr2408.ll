; RUN: llvm-as < %s | llc -march=c | grep {\\* ((unsigned int )}
; PR2408

define i32 @a(i32 %a) {
entry:
        %shr = ashr i32 %a, 0           ; <i32> [#uses=1]
        %shr2 = ashr i32 2, 0           ; <i32> [#uses=1]
        %mul = mul i32 %shr, %shr2              ; <i32> [#uses=1]
        %shr4 = ashr i32 2, 0           ; <i32> [#uses=1]
        %div = sdiv i32 %mul, %shr4             ; <i32> [#uses=1]
        ret i32 %div
}
