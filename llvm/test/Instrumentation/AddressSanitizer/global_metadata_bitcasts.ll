; Test that the compiler doesn't crash when the llvm.asan.globals containts
; an entry that points to a BitCast instruction.

; RUN: opt < %s -passes=asan -asan-globals-live-support=1 -S

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@g = global [1 x i32] zeroinitializer, align 4

!llvm.asan.globals = !{!0, !1}
!0 = !{[1 x i32]* @g, null, !"name", i1 false, i1 false}
!1 = !{i8* bitcast ([1 x i32]* @g to i8*), null, !"name", i1 false, i1 false}
