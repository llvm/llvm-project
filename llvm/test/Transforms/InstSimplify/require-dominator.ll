; RUN: opt < %s -passes=instsimplify

; instsimplify pass should explicitly require DominatorTreeAnalysis
; This test will segfault if DominatorTree is not available

target triple = "x86_64-grtev4-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo(i16 *) #1 align 2 {
  br i1 undef, label %exit, label %2

; <label>:2:
  %3 = tail call i8* @_Znwm(i64 56) #10
  %4 = bitcast i8* %3 to i16*
  %p = load i16*, i16** undef, align 8
  %5 = icmp eq i16* %p, %4
  br i1 %5, label %exit, label %6

; <label>:6:
  %7 = icmp eq i16* %p, null
  br i1 %7, label %exit, label %8

; <label>:8:
  br label %exit

exit:
  ret void
}

; Function Attrs: nobuiltin
declare i8* @_Znwm(i64)
