; RUN: llc -mtriple=x86_64 -relocation-model=static < %s | FileCheck --check-prefixes=CHECK,STATIC %s
; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,PIC %s
; RUN: llc -mtriple=x86_64 -code-model=medium -relocation-model=static < %s | FileCheck --check-prefixes=CHECK,MSTATIC %s
; RUN: llc -mtriple=x86_64 -code-model=medium -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,MPIC %s

@foo = internal global i32 0

define dso_local i64 @zero() {
; CHECK-LABEL: zero:
; CHECK:       # %bb.0:
; STATIC-NEXT:   movl $foo, %eax
; STATIC-NEXT:   retq
; PIC-NEXT:      leaq foo(%rip), %rax
; PIC-NEXT:      retq
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  retq
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rcx
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rax
; MPIC-NEXT:     addq %rcx, %rax
entry:
  ret i64 add (i64 ptrtoint (i32* @foo to i64), i64 0)
}

;; Check we don't fold a large offset into leaq, otherwise
;; the large r_addend can easily cause a relocation overflow.
define dso_local i64 @large() {
; CHECK-LABEL: large:
; CHECK:       # %bb.0:
; STATIC-NEXT:   movl $1701208431, %eax
; STATIC-NEXT:   leaq foo(%rax), %rax
; PIC-NEXT:      leaq foo(%rip), %rax
; PIC-NEXT:      addq $1701208431, %rax
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  addq $1701208431, %rax
; MSTATIC-NEXT:  retq
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rax
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rcx
; MPIC-NEXT:     leaq 1701208431(%rax,%rcx), %rax
entry:
  ret i64 add (i64 ptrtoint (i32* @foo to i64), i64 1701208431)
}
