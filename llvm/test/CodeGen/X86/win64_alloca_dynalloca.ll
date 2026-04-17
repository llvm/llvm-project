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
; W64:       $4144, %eax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: leaq 128(%rsp), %rbp

; Use %r11 for the large model.
; L64:       $4144, %eax
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

; W64: movabsq $15, %rax
; W64: addq  %rcx, %rax
; W64: andq  $-16, %rax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: leaq  48(%rsp), %rax

; L64: movabsq $15, %rax
; L64: addq  %rcx, %rax
; L64: andq  $-16, %rax
; L64: movabsq $__chkstk, %r11
; L64: callq *%r11
; L64: subq  %rax, %rsp
; L64: leaq  48(%rsp), %rax

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

; W64: movq  %rax, 32(%rsp)
; W64: leaq  -80(%rbp), %r9
; W64: callq bar

; EFI: subq  $48, %rsp
; EFI: movq  [[R64]], 32(%rsp)
; EFI: leaq  -[[B0OFS]](%rbp), %r9
; EFI: callq _bar

  ret i64 %r

; M64: leaq    3968(%rbp), %rsp

; W64: leaq    4016(%rbp), %rsp

}

define i64 @aligned(i64 %n, i64 %x, ptr %dummy) nounwind {
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

; W64: movabsq $142, %rax
; W64: addq  %rcx, %rax
; W64: andq  $-16, %rax
; W64: callq __chkstk
; W64: subq  %rax, %rsp
; W64: leaq  175(%rsp), [[R2:%r.*]]
; W64: andq  $-128, [[R2]]

; EFI: leaq  15(%{{.*}}), [[R1:%r.*]]
; EFI: andq  $-16, [[R1]]
; EFI: movq  %rsp, [[R64:%r.*]]
; EFI: subq  [[R1]], [[R64]]
; EFI: andq  $-128, [[R64]]
; EFI: movq  [[R64]], %rsp

  %r = call i64 @bar(i64 %n, i64 %x, i64 %n, ptr %dummy, ptr %buf1) nounwind

; M64: subq  $48, %rsp
; M64: movq  [[R2]], 32(%rsp)
; M64: callq bar

; W64: movq  [[R2]], 32(%rsp)
; W64: callq bar

; EFI: subq  $48, %rsp
; EFI: movq  [[R64]], 32(%rsp)
; EFI: callq _bar

  ret i64 %r
}

define void @two_allocas(i64 %a, i64 %b) nounwind {
; W64-LABEL: two_allocas:
; W64:       movabsq $15, %rax
; W64-NEXT:  addq %rcx, %rax
; W64-NEXT:  andq $-16, %rax
; W64-NEXT:  callq __chkstk
; W64-NEXT:  subq %rax, %rsp
; W64-NEXT:  leaq 32(%rsp), [[P:%r[a-z0-9]+]]
; W64:       movq [[P]], %rcx
; W64-NEXT:  callq use
; W64:       movabsq $15, %rax
; W64-NEXT:  addq %rsi, %rax
; W64-NEXT:  andq $-16, %rax
; W64-NEXT:  callq __chkstk
; W64-NEXT:  subq %rax, %rsp
; W64-NEXT:  leaq 32(%rsp), [[Q:%r[a-z0-9]+]]
; W64:       movq [[P]], %rcx
; W64-NEXT:  callq use
; W64-NEXT:  movq [[Q]], %rcx
; W64-NEXT:  callq use
entry:
  %p = alloca i8, i64 %a, align 1
  call void @use(ptr %p)
  %q = alloca i8, i64 %b, align 1
  call void @use(ptr %p)
  call void @use(ptr %q)
  ret void
}

define void @two_allocas_aligned(i64 %a, i64 %b) nounwind {
; W64-LABEL: two_allocas_aligned:
; W64:       movabsq $142, %rax
; W64-NEXT:  addq %rcx, %rax
; W64-NEXT:  andq $-16, %rax
; W64-NEXT:  callq __chkstk
; W64-NEXT:  subq %rax, %rsp
; W64-NEXT:  leaq 159(%rsp), [[P:%r[a-z0-9]+]]
; W64-NEXT:  andq $-128, [[P]]
; W64:       movabsq $15, %rax
; W64-NEXT:  addq %rdx, %rax
; W64-NEXT:  andq $-16, %rax
; W64-NEXT:  callq __chkstk
; W64-NEXT:  subq %rax, %rsp
; W64-NEXT:  leaq 32(%rsp), [[Q:%r[a-z0-9]+]]
; W64:       movq [[P]], %rcx
; W64-NEXT:  callq use
; W64-NEXT:  movq [[Q]], %rcx
; W64-NEXT:  callq use
; W64-NEXT:  movq [[P]], %rcx
; W64-NEXT:  callq use
entry:
  %p = alloca i8, i64 %a, align 128
  %q = alloca i8, i64 %b, align 1
  call void @use(ptr %p)
  call void @use(ptr %q)
  call void @use(ptr %p)
  ret void
}

define i64 @aligned_stack_args(i64 %n, i64 %x, ptr %dummy) nounwind {
; W64-LABEL: aligned_stack_args:
; W64:       movabsq $78, %rax
; W64-NEXT:  addq  %rcx, %rax
; W64-NEXT:  andq  $-16, %rax
; W64-NEXT:  callq __chkstk
; W64-NEXT:  subq  %rax, %rsp
; W64-NEXT:  leaq  111(%rsp), [[P:%r[a-z0-9]+]]
; W64-NEXT:  andq  $-64, [[P]]
; W64:       movq  [[P]], 40(%rsp)
; W64-NEXT:  movq  [[P]], 32(%rsp)
; W64-NOT:   subq
; W64-NOT:   addq
; W64:       callq baz
entry:
  %p = alloca i8, i64 %n, align 64
  %r = call i64 @baz(i64 %n, i64 %x, i64 %n, ptr %dummy, ptr %p, ptr %p) nounwind
  ret i64 %r
}

declare i64 @bar(i64, i64, i64, ptr nocapture, ptr nocapture) nounwind
declare i64 @baz(i64, i64, i64, ptr nocapture, ptr nocapture, ptr nocapture) nounwind
declare void @use(ptr) nounwind
