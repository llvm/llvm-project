; RUN: llc -mtriple=x86_64 -relocation-model=static < %s | FileCheck --check-prefixes=CHECK,STATIC %s
; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,PIC %s
; RUN: llc -mtriple=x86_64 -code-model=medium -relocation-model=static < %s | FileCheck --check-prefixes=CHECK,MSTATIC %s
; RUN: llc -mtriple=x86_64 -code-model=medium -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,MPIC %s

@foo = internal global i32 0

define dso_local i64 @zero() #0 {
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
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 0)
}

define dso_local i64 @one() #0 {
; CHECK-LABEL: one:
; CHECK:       # %bb.0:
; STATIC-NEXT:   movl $foo+1, %eax
; PIC-NEXT:      leaq foo+1(%rip), %rax
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  incq %rax
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rax
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rcx
; MPIC-NEXT:     leaq 1(%rax,%rcx), %rax
entry:
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 1)
}

;; Check we don't fold a large offset into leaq, otherwise
;; the large r_addend can easily cause a relocation overflow.
define dso_local i64 @large() #0 {
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
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 1701208431)
}

;; Test we don't emit movl foo-1, %eax. ELF R_X86_64_32 does not allow
;; a negative value.
define dso_local i64 @neg_1() #0 {
; CHECK-LABEL: neg_1:
; CHECK:       # %bb.0:
; STATIC-NEXT:   leaq foo-1(%rip), %rax
; PIC-NEXT:      leaq foo-1(%rip), %rax
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  decq %rax
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rax
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rcx
; MPIC-NEXT:     leaq -1(%rax,%rcx), %rax
entry:
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 -1)
}

;; Test we don't emit movl foo-2147483648, %eax. ELF R_X86_64_32 does not allow
;; a negative value.
define dso_local i64 @neg_0x80000000() #0 {
; CHECK-LABEL: neg_0x80000000:
; CHECK:       # %bb.0:
; STATIC-NEXT:   leaq foo-2147483648(%rip), %rax
; PIC-NEXT:      leaq foo-2147483648(%rip), %rax
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  addq $-2147483648, %rax
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rax
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rcx
; MPIC-NEXT:     leaq -2147483648(%rax,%rcx), %rax
entry:
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 -2147483648)
}

define dso_local i64 @neg_0x80000001() #0 {
; CHECK-LABEL: neg_0x80000001:
; CHECK:       # %bb.0:
; STATIC-NEXT:   movabsq $-2147483649, %rax
; STATIC-NEXT:   leaq foo(%rax), %rax
; PIC-NEXT:      leaq foo(%rip), %rcx
; PIC-NEXT:      movabsq $-2147483649, %rax
; PIC-NEXT:      addq %rcx, %rax
; MSTATIC-NEXT:  movabsq $-2147483649, %rcx
; MSTATIC-NEXT:  movabsq $foo, %rax
; MSTATIC-NEXT:  addq %rcx, %rax
; MPIC-NEXT:     leaq _GLOBAL_OFFSET_TABLE_(%rip), %rcx
; MPIC-NEXT:     movabsq $foo@GOTOFF, %rdx
; MPIC-NEXT:     movabsq $-2147483649, %rax
; MPIC-NEXT:     addq %rdx, %rax
; MPIC-NEXT:     addq %rcx, %rax
entry:
  ret i64 add (i64 ptrtoint (ptr @foo to i64), i64 -2147483649)
}

attributes #0 = { nounwind }
