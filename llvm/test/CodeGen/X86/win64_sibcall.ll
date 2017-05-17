; RUN: llc < %s -mtriple=x86_64-pc-win32-coreclr | FileCheck %s -check-prefix=WIN_X64
; RUN: llc < %s -mtriple=x86_64-pc-linux         | FileCheck %s -check-prefix=LINUX

%Object = type <{ [0 x i64*]* }>

define void @C1(%Object addrspace(1)* %param0) gc "coreclr" {
entry:

; WIN_X64: # BB#0:
; WIN_X64:	pushq	%rax
; LINUX:   # BB#0:                                 # %entry
; LINUX:	movq	$0, -8(%rsp)

  %this = alloca %Object addrspace(1)*
  store volatile %Object addrspace(1)* null, %Object addrspace(1)** %this
  store volatile %Object addrspace(1)* %param0, %Object addrspace(1)** %this
  br label %0

; <label>:0                                       ; preds = %entry
  %1 = load %Object addrspace(1)*, %Object addrspace(1)** %this, align 8

; WIN_X64:	xorl	%r8d, %r8d
; WIN_X64:	popq	%rax
; WIN_X64:	jmp	  C2                  # TAILCALL
; LINUX:	xorl	%edx, %edx
; LINUX:	jmp	C2                      # TAILCALL

  tail call void @C2(%Object addrspace(1)* %1, i32 0, %Object addrspace(1)* null)
  ret void
}

declare void @C2(%Object addrspace(1)*, i32, %Object addrspace(1)*)

; Function Attrs: nounwind
declare void @llvm.localescape(...) #0

attributes #0 = { nounwind }

