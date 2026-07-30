; RUN: llc < %s -mtriple=x86_64-pc-win32-coreclr | FileCheck %s -check-prefix=WIN_X64
; RUN: llc < %s -mtriple=x86_64-pc-linux         | FileCheck %s -check-prefix=LINUX

%Object = type <{ ptr }>

define void @C1(ptr addrspace(1) %param0) gc "coreclr" {
entry:

; WIN_X64: # %bb.0:
; WIN_X64:	pushq	%rax
; LINUX:   # %bb.0:                                 # %entry
; LINUX:	movq	$0, -8(%rsp)

  %this = alloca ptr addrspace(1)
  store volatile ptr addrspace(1) null, ptr %this
  store volatile ptr addrspace(1) %param0, ptr %this
  br label %0

; <label>:0                                       ; preds = %entry
  %1 = load ptr addrspace(1), ptr %this, align 8

; WIN_X64:	xorl	%r8d, %r8d
; WIN_X64:	popq	%rax
; WIN_X64:	jmp	  C2                  # TAILCALL
; LINUX:	xorl	%edx, %edx
; LINUX:	jmp	C2                      # TAILCALL

  tail call void @C2(ptr addrspace(1) %1, i32 0, ptr addrspace(1) null)
  ret void
}

declare dso_local void @C2(ptr addrspace(1), i32, ptr addrspace(1))

; Function Attrs: nounwind
declare dso_local void @llvm.localescape(...) #0

attributes #0 = { nounwind }

