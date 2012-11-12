; RUN: llc < %s
; PR2924

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define x86_stdcallcc { i32, i8* } @_D3std6string7toupperFAaZAa({ i32, i8* } %s) {
entry_std.string.toupper:
        %tmp58 = load i32* null
        %tmp59 = icmp eq i32 %tmp58, 0
        %r.val = load { i32, i8* }* null, align 8
        %condtmp.0 = select i1 %tmp59, { i32, i8* } undef, { i32, i8* } %r.val 

        ret { i32, i8* } %condtmp.0
}
define { } @empty({ } %s) {
entry_std.string.toupper:
        %tmp58 = load i32* null
        %tmp59 = icmp eq i32 %tmp58, 0
        %r.val = load { }* null, align 8
        %condtmp.0 = select i1 %tmp59, { } undef, { } %r.val
        ret { } %condtmp.0
}
