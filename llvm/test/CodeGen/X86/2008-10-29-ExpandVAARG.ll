; RUN: llc < %s -mtriple=i686--
; PR2977
define i8* @ap_php_conv_p2(){
entry:
        %ap.addr = alloca i8*           ; <i8**> [#uses=36]
        br label %sw.bb301
sw.bb301:
        %0 = va_arg i8** %ap.addr, i64          ; <i64> [#uses=1]
        br label %sw.bb301
}
