; RUN: llc -mtriple=x86_64-windows-msvc %s -o - -verify-machineinstrs | FileCheck %s -check-prefix=WINDOWS
; RUN: llc -mtriple=x86_64-linux-gnu    %s -o - -verify-machineinstrs | FileCheck %s -check-prefix=LINUX

declare void @h(ptr, i64, ptr)

define tailcc void @tailcall_frame(ptr %0, i64 %1) sspreq {
; WINDOWS-LABEL: tailcall_frame:
; WINDOWS: movq	__security_cookie(%rip), %rax
; WINDOWS: xorq	%rsp, %rax
; WINDOWS: movq	%rax, {{[0-9]*}}(%rsp)
; WINDOWS: movq	 {{[0-9]*}}(%rsp), %rcx
; WINDOWS: xorq	%rsp, %rcx
; WINDOWS: movq	__security_cookie(%rip), %rax
; WINDOWS: cmpq	%rcx, %rax
; WINDOWS: jne	.LBB0_1
; WINDOWS: xorl %ecx, %ecx
; WINDOWS: jmp h
; WINDOWS: .LBB0_1
; WINDOWS: callq __security_check_cookie
; WINDOWS: int3


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
; WINDOWS: movq	__security_cookie(%rip), %rax
; WINDOWS: xorq	%rsp, %rax
; WINDOWS: callq bar
; WINDOWS: movq	 {{[0-9]*}}(%rsp), %rcx
; WINDOWS: xorq	%rsp, %rcx
; WINDOWS: movq	__security_cookie(%rip), %rax
; WINDOWS: cmpq	%rcx, %rax
; WINDOWS: jne	.LBB1_1
; WINDOWS: jmp	bar
; WINDOWS: .LBB1_1
; WINDOWS: callq	__security_check_cookie
; WINDOWS: int3

; LINUX-LABEL: tailcall_unrelated_frame:
; LINUX: callq bar
; LINUX: jne
; LINUX: jmp bar
; LINUX: callq __stack_chk_fail

  call void @bar()
  tail call void @bar()
  ret void
}
