; RUN: not --crash llc -mtriple=spirv64-unknown-unknown -filetype=null %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=spirv32-unknown-unknown -filetype=null %s 2>&1 | FileCheck %s

; SPIRV has no STACK_CHECK_GUARD libcall impl, so getSDagStackGuard() returns
; nullptr. The stack protector check must report a failure to translate instead
; of dereferencing the null guard value.

; CHECK: error: unable to lower stackguard
; CHECK: LLVM ERROR: unable to translate basic block

define void @func() sspreq nounwind {
  %alloca = alloca i32, align 4
  call void @capture(ptr %alloca)
  ret void
}

declare void @capture(ptr)
