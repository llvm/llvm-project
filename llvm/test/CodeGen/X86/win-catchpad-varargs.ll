; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s --check-prefix=X86

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare i32 @__CxxFrameHandler3(...)
declare void @g()

define i32 @f(i32 %a, ...) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %ap = alloca i8*
  invoke void @g()
          to label %return unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %argp.cur = load i8*, i8** %ap
  %1 = bitcast i8* %argp.cur to i32*
  %arg2 = load i32, i32* %1
  call void @llvm.va_end(i8* %ap1)
  catchret %0 to label %return

catchendblock:                                    ; preds = %catch.dispatch
  catchendpad unwind to caller

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ %arg2, %catch ], [ -1, %entry ]
  ret i32 %retval.0
}

; X64-LABEL: .seh_proc f
; X64: pushq %rbp
; X64: pushq %rsi
; X64: subq $56, %rsp
; X64: leaq 48(%rsp), %rbp
; X64: movq $-2, (%rbp)
; X64: callq g
; X64: movl %esi, %eax
; X64: addq $56, %rsp
; X64: popq %rsi
; X64: popq %rbp

; X64: movl -4(%rbp), %esi
; X64: jmp

; X64-LABEL: "?catch$1@?0?f@4HA":
; X64: .seh_proc "?catch$1@?0?f@4HA"
; X64:         movq    %rdx, 16(%rsp)
; X64:         pushq   %rbp
; X64:         pushq   %rsi
; X64:         subq    $40, %rsp
; X64:         leaq    48(%rdx), %rbp
; arg2 is at RBP+40:
; start at arg2
; + 8 for arg1
; + 8 for retaddr
; + 8 for RBP
; + 8 for RSI
; + 56 for stackalloc
; - 48 for setframe
; = 40
; X64:         movl    40(%rbp), %eax
; X64:         movl    %eax, -4(%rbp)
; X64:         leaq    .LBB0_2(%rip), %rax
; X64:         addq    $40, %rsp
; X64: 	       popq    %rsi
; X64:         popq    %rbp
; X64:         retq                            # CATCHRET

; X86-LABEL: _f:                                     # @f
; X86:         pushl   %ebp
; X86:         movl    %esp, %ebp
; X86:         pushl   %ebx
; X86:         pushl   %edi
; X86:         pushl   %esi
; X86:         subl    $28, %esp
; X86: 	       movl    $-1, -40(%ebp)
; X86:         calll   _g
; X86:         movl    -40(%ebp), %eax
; X86:         addl    $28, %esp
; X86:         popl    %esi
; X86:         popl    %edi
; X86:         popl    %ebx
; X86:         popl    %ebp
; X86:         retl

; X86-LABEL: "?catch$1@?0?f@4HA":
; X86:         pushl   %ebp
; X86:         addl    $12, %ebp
; arg2 is at EBP offset 12:
; + 4 for arg1
; + 4 for retaddr
; + 4 for EBP
; Done due to mov %esp, %ebp
; X86:         movl    12(%ebp), %eax
; X86:         movl    %eax, -32(%ebp)
; X86:         movl    $LBB0_2, %eax
; X86:         popl    %ebp
; X86:         retl                            # CATCHRET
