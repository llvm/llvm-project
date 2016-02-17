; RUN: opt -prune-eh -S < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

declare void @neverthrows() nounwind

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
  invoke void @neverthrows()
          to label %try.cont unwind label %cleanuppad

try.cont:
  ret void

cleanuppad:
  %cp = cleanuppad within none []
  br label %cleanupret

cleanupret:
  cleanupret from %cp unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: call void @neverthrows()

; CHECK: %[[cp:.*]] = cleanuppad within none []
; CHECK-NEXT: unreachable

; CHECK: cleanupret from %[[cp]] unwind to caller

declare i32 @__CxxFrameHandler3(...)
