; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v OK | not grep add

;; Target triple for gep raising case below.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define i64 @test1(i64 %A, i32 %B) {
        %tmp12 = zext i32 %B to i64
        %tmp3 = shl i64 %tmp12, 32
        %tmp5 = add i64 %tmp3, %A
        %tmp6 = and i64 %tmp5, 123
        ret i64 %tmp6
}

; PR1795
define void @test2(i32 %.val24) {
EntryBlock:
        add i32 %.val24, -12
        inttoptr i32 %0 to i32*
        store i32 1, i32* %1
        add i32 %.val24, -16
        inttoptr i32 %2 to i32*
        getelementptr i32* %3, i32 1
        load i32* %4
        tail call i32 @callee( i32 %5 )
        ret void
}

declare i32 @callee(i32)


define i32 @test3(i32 %A) {
  %B = and i32 %A, 7
  %C = and i32 %A, 32
  %F = add i32 %B, %C
  ret i32 %F
}

define i32 @test4(i32 %A) {
  %B = and i32 %A, 128
  %C = lshr i32 %A, 30
  %F = add i32 %B, %C
  ret i32 %F
}

