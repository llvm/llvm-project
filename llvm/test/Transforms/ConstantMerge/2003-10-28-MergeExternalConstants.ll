; RUN: opt -S -constmerge %s | FileCheck %s

; CHECK: @foo = constant i32 6
; CHECK: @bar = constant i32 6
@foo = constant i32 6           ; <i32*> [#uses=0]
@bar = constant i32 6           ; <i32*> [#uses=0]

