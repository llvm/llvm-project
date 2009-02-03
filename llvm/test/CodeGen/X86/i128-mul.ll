; RUN: llvm-as < %s | llc -march=x86-64
; PR1198

define i64 @foo(i64 %x, i64 %y) {
        %tmp0 = zext i64 %x to i128
        %tmp1 = zext i64 %y to i128
        %tmp2 = mul i128 %tmp0, %tmp1
        %tmp7 = zext i32 64 to i128
        %tmp3 = lshr i128 %tmp2, %tmp7
        %tmp4 = trunc i128 %tmp3 to i64
        ret i64 %tmp4
}
