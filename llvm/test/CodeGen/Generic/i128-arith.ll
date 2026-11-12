; RUN: llc < %s 

define i64 @foo(i64 %x, i64 %y, i32 %amt) {
        %tmp0 = zext i64 %x to i128
        %tmp1 = sext i64 %y to i128
        %tmp2 = or i128 %tmp0, %tmp1
        %tmp7 = zext i32 13 to i128
        %tmp3 = lshr i128 %tmp2, %tmp7
        %tmp4 = trunc i128 %tmp3 to i64
        ret i64 %tmp4
}
