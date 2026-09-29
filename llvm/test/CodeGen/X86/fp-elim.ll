; RUN: llc < %s -mtriple=i686-- -asm-verbose=false                           | FileCheck %s --check-prefix=FP-ELIM
; RUN: llc < %s -mtriple=i686-- -asm-verbose=false -frame-pointer=all        | FileCheck %s --check-prefix=NO-ELIM

; Implement -momit-leaf-frame-pointer
; rdar://7886181

define i32 @t1() nounwind readnone {
entry:
; FP-ELIM-LABEL:  t1:
; FP-ELIM-NEXT:     movl
; FP-ELIM-NEXT:     ret

; NO-ELIM-LABEL:  t1:
; NO-ELIM-NEXT:     pushl %ebp
; NO-ELIM:          popl %ebp
; NO-ELIM-NEXT:     ret
  ret i32 10
}

define void @t2() nounwind {
entry:
; FP-ELIM-LABEL:  t2:
; FP-ELIM-NOT:      pushl %ebp
; FP-ELIM:          ret

; NO-ELIM-LABEL:  t2:
; NO-ELIM-NEXT:     pushl %ebp
; NO-ELIM:          popl %ebp
; NO-ELIM-NEXT:     ret
  tail call void @foo(i32 0) nounwind
  ret void
}

; The local non-leaf attribute takes precendece over the command line flag.
define i32 @t3() "frame-pointer"="non-leaf" nounwind readnone {
entry:
; ANY-ELIM-LABEL:  t3:
; ANY-ELIM-NEXT:     movl
; ANY-ELIM-NEXT:     ret
  ret i32 10
}

define void @t4() "frame-pointer"="non-leaf" nounwind {
entry:
; ANY-ELIM-LABEL:  t4:
; ANY-ELIM-NEXT:     pushl %ebp
; ANY-ELIM:          popl %ebp
; ANY-ELIM-NEXT:     ret
  tail call void @foo(i32 0) nounwind
  ret void
}

declare void @foo(i32)
