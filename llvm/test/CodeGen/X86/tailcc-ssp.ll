; RUN: llc -mtriple=x86_64-windows-msvc %s -o - -verify-machineinstrs | FileCheck %s -check-prefix=WINDOWS
; RUN: llc -mtriple=x86_64-linux-gnu    %s -o - -verify-machineinstrs | FileCheck %s -check-prefix=LINUX

declare void @h(ptr, i64, ptr)

define tailcc void @tailcall_frame(ptr %0, i64 %1) sspreq {
; WINDOWS-LABEL: tailcall_frame:
; WINDOWS: callq __security_check_cookie
; WINDOWS: xorl %ecx, %ecx
; WINDOWS: jmp h

; LINUX-LABEL: tailcall_frame:
; LINUX: jne
; LINUX: jmp h
; LINUX: callq __stack_chk_fail

   tail call tailcc void @h(ptr null, i64 0, ptr null)
   ret void
}

declare void @bar()
define void @tailcall_unrelated_frame() sspreq {
; WINDOWS-LABEL: tailcall_unrelated_frame:
; WINDOWS: subq [[STACK:\$.*]], %rsp
; WINDOWS: callq bar
; WINDOWS: callq __security_check_cookie
; WINDOWS: addq [[STACK]], %rsp
; WINDOWS: jmp bar

; LINUX-LABEL: tailcall_unrelated_frame:
; LINUX: callq bar
; LINUX: jne
; LINUX: jmp bar
; LINUX: callq __stack_chk_fail

  call void @bar()
  tail call void @bar()
  ret void
}
