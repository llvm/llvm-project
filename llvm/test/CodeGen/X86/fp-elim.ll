; RUN: llc < %s -march=x86 -asm-verbose=false                           | FileCheck %s -check-prefix=FP-ELIM
; RUN: llc < %s -march=x86 -asm-verbose=false -disable-fp-elim          | FileCheck %s -check-prefix=NO-ELIM
; RUN: llc < %s -march=x86 -asm-verbose=false -disable-non-leaf-fp-elim | FileCheck %s -check-prefix=NON-LEAF

; Implement -momit-leaf-frame-pointer
; rdar://7886181

define i32 @t1() nounwind readnone {
entry:
; FP-ELIM:      t1:
; FP-ELIM-NEXT: movl
; FP-ELIM-NEXT: ret

; NO-ELIM:      t1:
; NO-ELIM-NEXT: pushl %ebp
; NO-ELIM:      popl %ebp
; NO-ELIM-NEXT: ret

; NON-LEAF:      t1:
; NON-LEAF-NEXT: movl
; NON-LEAF-NEXT: ret
  ret i32 10
}

define void @t2() nounwind {
entry:
; FP-ELIM:     t2:
; FP-ELIM-NOT: pushl %ebp
; FP-ELIM:     ret

; NO-ELIM:      t2:
; NO-ELIM-NEXT: pushl %ebp
; NO-ELIM:      popl %ebp
; NO-ELIM-NEXT: ret

; NON-LEAF:      t2:
; NON-LEAF-NEXT: pushl %ebp
; NON-LEAF:      popl %ebp
; NON-LEAF-NEXT: ret
  tail call void @foo(i32 0) nounwind
  ret void
}

declare void @foo(i32)
