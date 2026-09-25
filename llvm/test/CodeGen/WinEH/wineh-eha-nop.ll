; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: %if aarch64-registered-target %{ llc -mtriple=aarch64-pc-windows-msvc < %s | FileCheck %s --check-prefix=ARM64 %}

; x86 pads a faulting instruction that opens an EH region away from the return
; of the call before it. The Arm64 unwinder backs the PC up itself, so it does
; not need that nop.

declare i32 @__CxxFrameHandler3(...)
declare void @llvm.seh.scope.begin()
declare void @llvm.seh.scope.end()
declare void @dtor(ptr)

define void @f(ptr %p) personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @llvm.seh.scope.begin()
          to label %cont unwind label %cleanup

cont:
  store volatile i32 0, ptr %p, align 4
  invoke void @llvm.seh.scope.end()
          to label %done unwind label %cleanup

done:
  ret void

cleanup:
  %cp = cleanuppad within none []
  call void @dtor(ptr %p) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; X64-LABEL: f:
; X64:        # EH_LABEL
; X64-NEXT:   nop

; ARM64-LABEL: f:
; ARM64:       // EH_LABEL
; ARM64-NOT:   nop

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"eh-asynch", i32 1}
