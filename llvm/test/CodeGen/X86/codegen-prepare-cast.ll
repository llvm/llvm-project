; RUN: llc < %s
; PR4297
; RUN: opt -S < %s -codegenprepare | FileCheck %s

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
        %"byte[]" = type { i64, ptr }
        %"char[][]" = type { i64, ptr }
@.str = external dso_local constant [7 x i8]              ; <ptr> [#uses=1]

; CHECK-LABEL: @_Dmain
; CHECK: load i8, ptr %tmp4
; CHECK: ret
define fastcc i32 @_Dmain(%"char[][]" %unnamed) {
entry:
        %tmp = getelementptr [7 x i8], ptr @.str, i32 0, i32 0              ; <ptr> [#uses=1]
        br i1 undef, label %foreachbody, label %foreachend

foreachbody:            ; preds = %entry
        %tmp4 = getelementptr i8, ptr %tmp, i32 undef               ; <ptr> [#uses=1]
        %tmp5 = load i8, ptr %tmp4          ; <i8> [#uses=0]
        unreachable

foreachend:             ; preds = %entry
        ret i32 0
}

