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
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %argp.cur = load i8*, i8** %ap
  %1 = bitcast i8* %argp.cur to i32*
  %arg2 = load i32, i32* %1
  call void @llvm.va_end(i8* %ap1)
  catchret from %0 to label %return

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ %arg2, %catch ], [ -1, %entry ]
  ret i32 %retval.0
}

; X64-LABEL: .seh_proc f
; X64: pushq %rbp
; X64: subq $64, %rsp
; X64: leaq 64(%rsp), %rbp
; X64: movq $-2, -8(%rbp)
; X64: movl    $-1, -20(%rbp) # 4-byte Folded Spill
; X64: callq g
; X64: .LBB0_1
; X64: movl    -20(%rbp), %eax # 4-byte Reload
; X64: addq $64, %rsp
; X64: popq %rbp

; X64-LABEL: "?catch${{[0-9]}}@?0?f@4HA":
; X64: .seh_proc "?catch${{[0-9]}}@?0?f@4HA"
; X64:         movq    %rdx, 16(%rsp)
; X64:         pushq   %rbp
; X64:         subq    $32, %rsp
; X64:         leaq    64(%rdx), %rbp
; arg2 is at RBP+40:
; start at arg2
; + 8 for arg1
; + 8 for retaddr
; + 8 for RBP
; + 64 for stackalloc
; - 64 for setframe
; = 40
; X64:         movl    24(%rbp), %eax
; X64:         movl    %eax, -20(%rbp)  # 4-byte Spill
; X64:         leaq    .LBB0_1(%rip), %rax
; X64:         addq    $32, %rsp
; X64:         popq    %rbp
; X64:         retq                            # CATCHRET

; X86-LABEL: _f:                                     # @f
; X86:         pushl   %ebp
; X86:         movl    %esp, %ebp
; X86:         pushl   %ebx
; X86:         pushl   %edi
; X86:         pushl   %esi
; X86:         subl    $24, %esp
; X86: 	       movl    $-1, -36(%ebp)
; X86:         calll   _g
; X86: LBB0_[[retbb:[0-9]+]]:
; X86:         movl    -36(%ebp), %eax
; X86:         addl    $24, %esp
; X86:         popl    %esi
; X86:         popl    %edi
; X86:         popl    %ebx
; X86:         popl    %ebp
; X86:         retl

; X86: LBB0_[[restorebb:[0-9]+]]: # Block address taken
; X86: addl $12, %ebp
; arg2 is at EBP offset 12:
; + 4 for arg1
; + 4 for retaddr
; + 4 for EBP
; X86: movl 12(%ebp), %eax
; X86: movl %eax, -36(%ebp)
; X86: jmp LBB0_[[retbb]]

; X86-LABEL: "?catch${{[0-9]}}@?0?f@4HA":
; X86:         pushl   %ebp
; X86:         addl    $12, %ebp
; Done due to mov %esp, %ebp
; X86:         leal    12(%ebp), %eax
; X86:         movl    $LBB0_[[restorebb]], %eax
; X86:         popl    %ebp
; X86:         retl                            # CATCHRET
