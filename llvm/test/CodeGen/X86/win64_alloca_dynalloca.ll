; RUN: llc < %s -mcpu=generic -mtriple=x86_64-mingw32     | FileCheck %s -check-prefix=M64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-win32       | FileCheck %s -check-prefix=W64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-win32-macho | FileCheck %s -check-prefix=EFI
; PR8777
; PR8778

define i64 @foo(i64 %n, i64 %x) nounwind {
entry:

  %buf0 = alloca i8, i64 4096, align 1

; ___chkstk must adjust %rsp.
; M64: movq  %rsp, %rbp
; M64:       $4096, %rax
; M64: callq ___chkstk
; M64-NOT:   %rsp

; __chkstk does not adjust %rsp.
; W64: movq  %rsp, %rbp
; W64:       $4096, %rax
; W64: callq __chkstk
; W64: subq  $4096, %rsp

; Freestanding
; EFI: movq  %rsp, %rbp
; EFI:       $[[B0OFS:4096|4104]], %rsp
; EFI-NOT:   call

  %buf1 = alloca i8, i64 %n, align 1

; M64: leaq  15(%{{.*}}), %rax
; M64: andq  $-16, %rax
; M64: callq ___chkstk
; M64-NOT:   %rsp
; M64: movq  %rsp, %rax

; W64: leaq  15(%{{.*}}), %rax
; W64: andq  $-16, %rax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: movq  %rsp, %rax

; EFI: leaq  15(%{{.*}}), [[R1:%r.*]]
; EFI: andq  $-16, [[R1]]
; EFI: movq  %rsp, [[R64:%r.*]]
; EFI: subq  [[R1]], [[R64]]
; EFI: movq  [[R64]], %rsp

  %r = call i64 @bar(i64 %n, i64 %x, i64 %n, i8* %buf0, i8* %buf1) nounwind

; M64: subq  $48, %rsp
; M64: leaq  -4096(%rbp), %r9
; M64: movq  %rax, 32(%rsp)
; M64: callq bar

; W64: subq  $48, %rsp
; W64: leaq  -4096(%rbp), %r9
; W64: movq  %rax, 32(%rsp)
; W64: callq bar

; EFI: subq  $48, %rsp
; EFI: leaq  -[[B0OFS]](%rbp), %r9
; EFI: movq  [[R64]], 32(%rsp)
; EFI: callq _bar

  ret i64 %r

; M64: movq    %rbp, %rsp

; W64: movq    %rbp, %rsp

}

declare i64 @bar(i64, i64, i64, i8* nocapture, i8* nocapture) nounwind
