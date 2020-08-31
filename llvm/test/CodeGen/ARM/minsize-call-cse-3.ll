; RUN: llc < %s
; Verify that we don't crash on indirect function calls
; in Thumb1InstrInfo::foldMemoryOperand.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; Function Attrs: minsize nounwind optsize
define void @test(i32* %p, i32 %x, i32 %y, i32 %z) {
entry:
  tail call void inttoptr (i32 19088743 to void (i32*, i32, i32, i32)*)(i32* %p, i32 %x, i32 %y, i32 %z)
  tail call void inttoptr (i32 19088743 to void (i32*, i32, i32, i32)*)(i32* %p, i32 %x, i32 %y, i32 %z)
  tail call void inttoptr (i32 19088743 to void (i32*, i32, i32, i32)*)(i32* %p, i32 %x, i32 %y, i32 %z)
  tail call void inttoptr (i32 19088743 to void (i32*, i32, i32, i32)*)(i32* %p, i32 %x, i32 %y, i32 %z)
  ret void
}
