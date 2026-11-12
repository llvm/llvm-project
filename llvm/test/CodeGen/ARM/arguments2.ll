; RUN: llc < %s -mtriple=arm-linux-gnueabi
; RUN: llc < %s -mtriple=arm-apple-darwin

define i32 @f(i32 %a, i128 %b) {
        %tmp = call i32 @g(i128 %b)
        ret i32 %tmp
}

declare i32 @g(i128)
