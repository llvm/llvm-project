; RUN: llc < %s -mcpu=generic -enable-misched=false -mtriple=x86_64-mingw32     | FileCheck %s -check-prefix=M64
; RUN: llc < %s -mcpu=generic -enable-misched=false -mtriple=x86_64-win32       | FileCheck %s -check-prefix=W64
; RUN: llc < %s -mcpu=generic -enable-misched=false -mtriple=x86_64-win32 -code-model=large | FileCheck %s -check-prefix=L64
; RUN: llc < %s -mcpu=generic -enable-misched=false -mtriple=x86_64-win32-macho | FileCheck %s -check-prefix=EFI
; PR8777
; PR8778

define i64 @unaligned(i64 %n, i64 %x) nounwind {
; M64-LABEL: unaligned:
; W64-LABEL: unaligned:
; EFI-LABEL: unaligned:
entry:

  %buf0 = alloca i8, i64 4096, align 1

; ___chkstk_ms does not adjust %rsp.
; M64:       $4096, %eax
; M64: callq ___chkstk_ms
; M64: subq  %rax, %rsp
; M64: leaq 128(%rsp), %rbp

; __chkstk does not adjust %rsp.
; W64:       $4096, %eax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: leaq 128(%rsp), %rbp

; Use %r11 for the large model.
; L64:       $4096, %eax
; L64: movabsq $__chkstk, %r11
; L64: callq *%r11
; L64: subq  %rax, %rsp

; Freestanding
; EFI:       $[[B0OFS:4096|4104]], %rsp
; EFI-NOT:   call

  %buf1 = alloca i8, i64 %n, align 1

; M64: leaq  15(%{{.*}}), %rax
; M64: andq  $-16, %rax
; M64: callq ___chkstk_ms
; M64: subq  %rax, %rsp
; M64: movq  %rsp, %rax

; W64: leaq  15(%{{.*}}), %rax
; W64: andq  $-16, %rax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: movq  %rsp, %rax

; L64: leaq  15(%{{.*}}), %rax
; L64: andq  $-16, %rax
; L64: movabsq $__chkstk, %r11
; L64: callq *%r11
; L64: subq  %rax, %rsp
; L64: movq  %rsp, %rax

; EFI: leaq  15(%{{.*}}), [[R1:%r.*]]
; EFI: andq  $-16, [[R1]]
; EFI: movq  %rsp, [[R64:%r.*]]
; EFI: subq  [[R1]], [[R64]]
; EFI: movq  [[R64]], %rsp

  %r = call i64 @bar(i64 %n, i64 %x, i64 %n, ptr %buf0, ptr %buf1) nounwind

; M64: subq  $48, %rsp
; M64: movq  %rax, 32(%rsp)
; M64: leaq  -128(%rbp), %r9
; M64: callq bar

; W64: subq  $48, %rsp
; W64: movq  %rax, 32(%rsp)
; W64: leaq  -128(%rbp), %r9
; W64: callq bar

; EFI: subq  $48, %rsp
; EFI: movq  [[R64]], 32(%rsp)
; EFI: leaq  -[[B0OFS]](%rbp), %r9
; EFI: callq _bar

  ret i64 %r

; M64: leaq    3968(%rbp), %rsp

; W64: leaq    3968(%rbp), %rsp

}

define i64 @aligned(i64 %n, i64 %x) nounwind {
; M64-LABEL: aligned:
; W64-LABEL: aligned:
; EFI-LABEL: aligned:
entry:

  %buf1 = alloca i8, i64 %n, align 128

; M64: leaq  15(%{{.*}}), %rax
; M64: andq  $-16, %rax
; M64: callq ___chkstk_ms
; M64: subq  %rax, %rsp
; M64: movq  %rsp, [[R2:%r.*]]
; M64: andq  $-128, [[R2]]
; M64: movq  [[R2]], %rsp

; W64: leaq  15(%{{.*}}), %rax
; W64: andq  $-16, %rax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: movq  %rsp, [[R2:%r.*]]
; W64: andq  $-128, [[R2]]
; W64: movq  [[R2]], %rsp

; EFI: leaq  15(%{{.*}}), [[R1:%r.*]]
; EFI: andq  $-16, [[R1]]
; EFI: movq  %rsp, [[R64:%r.*]]
; EFI: subq  [[R1]], [[R64]]
; EFI: andq  $-128, [[R64]]
; EFI: movq  [[R64]], %rsp

  %r = call i64 @bar(i64 %n, i64 %x, i64 %n, ptr undef, ptr %buf1) nounwind

; M64: subq  $48, %rsp
; M64: movq  [[R2]], 32(%rsp)
; M64: callq bar

; W64: subq  $48, %rsp
; W64: movq  [[R2]], 32(%rsp)
; W64: callq bar

; EFI: subq  $48, %rsp
; EFI: movq  [[R64]], 32(%rsp)
; EFI: callq _bar

  ret i64 %r
}

declare i64 @bar(i64, i64, i64, ptr nocapture, ptr nocapture) nounwind
