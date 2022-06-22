; RUN: llc -mtriple=x86_64-windows-msvc %s -o - -verify-machineinstrs | FileCheck %s

declare void @h(ptr, i64, ptr)

define tailcc void @tailcall_frame(ptr %0, i64 %1) sspreq {
; CHECK-LABEL: tailcall_frame:
; CHECK: callq __security_check_cookie
; CHECK: xorl %ecx, %ecx
; CHECK: jmp h

   tail call tailcc void @h(ptr null, i64 0, ptr null)
   ret void
}

declare void @bar()
define void @tailcall_unrelated_frame() sspreq {
; CHECK-LABEL: tailcall_unrelated_frame:
; CHECK: subq [[STACK:\$.*]], %rsp
; CHECK: callq bar
; CHECK: callq __security_check_cookie
; CHECK: addq [[STACK]], %rsp
; CHECK: jmp bar
  call void @bar()
  tail call void @bar()
  ret void
}
