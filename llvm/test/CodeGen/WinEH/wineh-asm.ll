; RUN: opt -winehprepare < %s

target triple = "x86_64-pc-windows-msvc"

define void @test1() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f(i32 1)
     to label %exit unwind label %cleanup

cleanup:
  %cp = cleanuppad within none []
  call void asm sideeffect "", ""()
  cleanupret from %cp unwind to caller

exit:
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK:      %[[cp:.*]] = cleanuppad within none []
; CHECK-NEXT: call void asm sideeffect "", ""()
; CHECK-NEXT: cleanupret from %[[cp]] unwind to caller

declare void @f(i32)

declare i32 @__CxxFrameHandler3(...)
