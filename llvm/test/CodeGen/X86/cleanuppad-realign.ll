; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck --check-prefix=X86 %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck --check-prefix=X64 %s

declare i32 @__CxxFrameHandler3(...)
declare void @Dtor(i64* %o)
declare void @f(i32)

define void @realigned_cleanup() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  ; Overalign %o to cause stack realignment.
  %o = alloca i64, align 32
  invoke void @f(i32 1)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call void @Dtor(i64* %o)
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad []
  call void @Dtor(i64* %o)
  cleanupret %0 unwind to caller
}

; X86-LABEL: _realigned_cleanup: # @realigned_cleanup
; X86:         pushl   %ebp
; X86:         movl    %esp, %ebp
; X86:         pushl   %ebx
; X86:         pushl   %edi
; X86:         pushl   %esi
; X86:         andl    $-32, %esp
; X86:         subl    $96, %esp
; X86:         movl    %esp, %esi
;	EBP will reload from this offset.
; X86:         movl    %ebp, 28(%esi)
; 	The last EH reg field is the state number, so dtor adjust is this +4.
; X86:         movl    $-1, 72(%esi)

; X86-LABEL: "?dtor$2@?0?realigned_cleanup@4HA":
; X86:         pushl   %ebp
; X86:         leal    -76(%ebp), %esi
; X86:         movl    28(%esi), %ebp
;	We used to have a bug where we clobbered ESI after the prologue.
; X86-NOT: 	movl {{.*}}, %esi
; X86:         popl    %ebp
; X86:         retl                            # CLEANUPRET

; X64-LABEL: realigned_cleanup: # @realigned_cleanup
; X64:         pushq   %rbp
; X64:         .seh_pushreg 5
; X64:         pushq   %rbx
; X64:         .seh_pushreg 3
; X64:         subq    $72, %rsp
; X64:         .seh_stackalloc 72
; X64:         leaq    64(%rsp), %rbp
; X64:         .seh_setframe 5, 64
; X64:         .seh_endprologue
; X64:         andq    $-32, %rsp
; X64:         movq    %rsp, %rbx
;	RBP will reload from this offset.
; X64:         movq    %rbp, 48(%rbx)

; X64-LABEL: "?dtor$2@?0?realigned_cleanup@4HA":
; X64:         movq    %rdx, 16(%rsp)
; X64:         pushq   %rbp
; X64:         .seh_pushreg 5
; X64:         pushq   %rbx
; X64:         .seh_pushreg 3
; X64:         subq    $40, %rsp
; X64:         .seh_stackalloc 40
; X64:         leaq    64(%rdx), %rbp
; X64:         .seh_endprologue
; X64: 	       andq    $-32, %rdx
; X64: 	       movq    %rdx, %rbx
; X64-NOT: 	mov{{.*}}, %rbx
; X64:         popq    %rbp
; X64:         retq                            # CLEANUPRET
