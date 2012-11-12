; RUN: llc < %s
; rdar://6781755
; PR3934

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-undermydesk-freebsd8.0"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
        %call = tail call i32 (...)* @getpid()          ; <i32> [#uses=1]
        %conv = trunc i32 %call to i16          ; <i16> [#uses=1]
        %0 = tail call i16 asm "xchgb ${0:h}, ${0:b}","=Q,0,~{dirflag},~{fpsr},~{flags}"(i16 %conv) nounwind           ; <i16> [#uses=0]
        ret i32 undef
}

declare i32 @getpid(...)
