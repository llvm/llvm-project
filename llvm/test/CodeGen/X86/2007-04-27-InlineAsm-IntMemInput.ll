; RUN: llc < %s | not grep "bsrl.*10"
; PR1356

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

define i32 @main() {
entry:
        %tmp4 = tail call i32 asm "bsrl  $1, $0", "=r,ro,~{dirflag},~{fpsr},~{flags},~{cc}"( i32 10 )           ; <i32> [#uses=1]
        ret i32 %tmp4
}

