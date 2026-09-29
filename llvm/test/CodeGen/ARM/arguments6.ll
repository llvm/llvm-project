; RUN: llc < %s -mtriple=arm-linux-gnueabi
; RUN: llc < %s -mtriple=arm-apple-darwin

define i128 @f(i32 %a, i128 %b) {
        %tmp = call i128 @g(i128 %b)
        ret i128 %tmp
}

declare i128 @g(i128)
