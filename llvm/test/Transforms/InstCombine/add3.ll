; RUN: opt < %s -passes=instcombine -S | grep inttoptr | count 2

;; Target triple for gep raising case below.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

; PR1795
define void @test2(i32 %.val24) {
EntryBlock:
        add i32 %.val24, -12
        inttoptr i32 %0 to ptr
        store i32 1, ptr %1
        add i32 %.val24, -16
        inttoptr i32 %2 to ptr
        getelementptr i32, ptr %3, i32 1
        load i32, ptr %4
        tail call i32 @callee( i32 %5 )
        ret void
}

declare i32 @callee(i32)
