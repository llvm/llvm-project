; RUN: llc < %s -mtriple=arm-linux-gnueabi
; RUN: llc < %s -mtriple=arm-apple-darwin

define i64 @f(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i64 %b) {
        %tmp = call i64 @g(i32 %a2, i32 %a3, i32 %a4, i32 %a5, i64 %b)
        ret i64 %tmp
}

declare i64 @g(i32, i32, i32, i32, i64)
