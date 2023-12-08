; RUN: opt < %s -passes=gvn -S | FileCheck %s
; PR2213

define ptr @f(ptr %x) {
entry:
        %tmp = call ptr @m( i32 12 )            ; <ptr> [#uses=2]
        %tmp1 = bitcast ptr %tmp to ptr                ; <ptr> [#uses=0]
        %tmp2 = bitcast ptr %tmp to ptr                ; <ptr> [#uses=0]
; CHECK-NOT: %tmp2
        ret ptr %tmp2
}

declare ptr @m(i32)
