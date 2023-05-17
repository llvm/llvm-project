; RUN: llc < %s | FileCheck %s

target triple = "i686--windows-msvc"

declare { ptr, i32 } @param2_ret2(i32, i32)
declare i32 @__CxxFrameHandler3(...)


define void @test_reserved_regs() minsize optsize personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: test_reserved_regs:
; CHECK: calll _param2_ret2
; CHECK-NEXT: popl %ecx
; CHECK-NEXT: popl %edi
start:
  %s = alloca i64
  store i64 4, ptr %s
  %0 = invoke { ptr, i32 } @param2_ret2(i32 0, i32 1)
          to label %out unwind label %cleanup

out:
  ret void

cleanup:
  %cp = cleanuppad within none []
  cleanupret from %cp unwind to caller
}
