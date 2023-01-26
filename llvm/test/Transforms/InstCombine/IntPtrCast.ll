; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:32:32"

define ptr @test(ptr %P) {
        %V = ptrtoint ptr %P to i32            ; <i32> [#uses=1]
        %P2 = inttoptr i32 %V to ptr           ; <ptr> [#uses=1]
        ret ptr %P2
; CHECK: ret ptr %P
}

