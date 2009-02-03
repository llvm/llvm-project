; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm -mattr=+v5TE
; RUN: llvm-as < %s | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smulbt | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smultt | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smlabt | count 1

@x = weak global i16 0          ; <i16*> [#uses=1]
@y = weak global i16 0          ; <i16*> [#uses=0]

define i32 @f1(i32 %y) {
        %tmp = load i16* @x             ; <i16> [#uses=1]
        %tmp1 = add i16 %tmp, 2         ; <i16> [#uses=1]
        %tmp2 = sext i16 %tmp1 to i32           ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp2, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f2(i32 %x, i32 %y) {
        %tmp1 = ashr i32 %x, 16         ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp3, %tmp1            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f3(i32 %a, i16 %x, i32 %y) {
        %tmp = sext i16 %x to i32               ; <i32> [#uses=1]
        %tmp2 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp3 = mul i32 %tmp2, %tmp             ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp3, %a               ; <i32> [#uses=1]
        ret i32 %tmp5
}

