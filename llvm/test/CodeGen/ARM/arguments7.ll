; RUN: llc < %s -mtriple=arm-linux-gnueabi
; RUN: llc < %s -mtriple=arm-apple-darwin

define double @f(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, double %b) {
        %tmp = call double @g(i32 %a2, i32 %a3, i32 %a4, i32 %a5, double %b)
        ret double %tmp
}

declare double @g(i32, i32, i32, i32, double)
