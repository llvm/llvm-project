; RUN: llc < %s -relocation-model=static | FileCheck %s
; RUN: llc < %s -relocation-model=pic | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llc < %s -relocation-model=pic -code-model=large | FileCheck %s --check-prefixes=CHECK,LARGE

; FIXME: Remove '-relocation-model=static' when it is no longer necessary to
; trigger the separate .rdata section.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

define void @f(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  tail call void @g(i32 0) #2
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  tail call void @g(i32 1) #2
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  tail call void @g(i32 2) #2
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  tail call void @g(i32 3) #2
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  tail call void @g(i32 10) #2
  ret void
}

declare void @g(i32)

; CHECK: .text
; CHECK: f:
; CHECK: .seh_proc f
; CHECK: .seh_endprologue

; STATIC: movl .LJTI0_0(,%rax,4), %eax
; STATIC: leaq __ImageBase(%rax), %rax
; STATIC: jmpq *%rax

; PIC: movl %ecx, %eax
; PIC: leaq .LJTI0_0(%rip), %rcx
; PIC: movl (%rcx,%rax,4), %eax
; PIC: leaq __ImageBase(%rip), %rcx
; PIC: addq %rax, %rcx
; PIC: jmpq *%rcx

; LARGE: movl %ecx, %eax
; LARGE-NEXT: movabsq $.LJTI0_0, %rcx
; LARGE-NEXT: movl (%rcx,%rax,4), %eax
; LARGE-NEXT: movabsq $__ImageBase, %rcx
; LARGE-NEXT: addq %rax, %rcx
; LARGE-NEXT: jmpq *%rcx

; CHECK: .LBB0_{{.*}}: # %sw.bb
; CHECK: .LBB0_{{.*}}: # %sw.bb2
; CHECK: .LBB0_{{.*}}: # %sw.bb3
; CHECK: .LBB0_{{.*}}: # %sw.bb1
; STATIC: callq g
; STATIC: jmp g # TAILCALL
; CHECK: .section        .rdata,"dr"
; CHECK: .LJTI0_0:
; CHECK: .long .LBB0_{{[0-9]+}}@IMGREL
; CHECK: .long .LBB0_{{[0-9]+}}@IMGREL
; CHECK: .long .LBB0_{{[0-9]+}}@IMGREL
; CHECK: .long .LBB0_{{[0-9]+}}@IMGREL

; It's important that we switch back to .text here, not .rdata.
; CHECK: .text
; CHECK: .seh_endproc
