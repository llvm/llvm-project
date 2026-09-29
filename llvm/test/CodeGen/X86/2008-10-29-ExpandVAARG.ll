; RUN: llc < %s -mtriple=i686--
; PR2977
define ptr @ap_php_conv_p2(){
entry:
        %ap.addr = alloca ptr           ; <ptr> [#uses=36]
        br label %sw.bb301
sw.bb301:
        %0 = va_arg ptr %ap.addr, i64          ; <i64> [#uses=1]
        br label %sw.bb301
}
