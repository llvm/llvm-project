; RUN: llc < %s -march=thumb

define i64 @f0(i64 %A, i64 %B) {
        %tmp = bitcast i64 %A to i64
        %tmp2 = lshr i64 %B, 1
        %tmp3 = sub i64 %tmp, %tmp2
        ret i64 %tmp3
}

define i32 @f1(i64 %x, i64 %y) {
        %a = shl i64 %x, %y
        %b = trunc i64 %a to i32
        ret i32 %b
}

define i32 @f2(i64 %x, i64 %y) {
        %a = ashr i64 %x, %y
        %b = trunc i64 %a to i32
        ret i32 %b
}

define i32 @f3(i64 %x, i64 %y) {
        %a = lshr i64 %x, %y
        %b = trunc i64 %a to i32
        ret i32 %b
}
