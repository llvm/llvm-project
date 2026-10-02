; RUN: llc -mtriple=aarch64-unknown-windows-msvc -filetype=obj < %s | llvm-objdump -d - | FileCheck %s

; Check that there is no instruction size verification error when a nop has
; to be inserted for an EH_LABEL.

; CHECK: nop
define void @test() personality ptr @__C_specific_handler {
entry:
  %a = alloca i32, align 4
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:
  %a.val = load volatile i32, ptr %a, align 4
  invoke void @llvm.seh.try.end()
          to label %exit unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:
  %cp = catchpad within %cs [ptr @filt]
  catchret from %cp to label %exit

exit:
  ret void
}

declare i32 @filt(ptr %exception_pointers, ptr %frame_pointer)

declare i32 @__C_specific_handler(...)

!llvm.module.flags = !{!1}
!1 = !{i32 2, !"eh-asynch", i32 1}
