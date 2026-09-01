; RUN: llc -mtriple=x86_64 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -O2 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -O0 -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s

define i32 @identity_i32(i32 %x) {
  %y = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
  ret i32 %y
}

; CHECK-LABEL: identity_i32:
; CHECK-NOT:   call
; CHECK:       movl %edi, %eax
; CHECK-NOT:   call
; CHECK-NEXT:  retq

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

!0 = !{!"amdgpu.vgpr"}
