; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=CHECK32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=CHECK64

define i32 @addRR(i32 %x, i32 %y) {
; CHECK32-LABEL: addRR:
; CHECK32:       # %bb.0: # %entry
; CHECK32-NEXT:    add.w $a0, $a1, $a0
; CHECK32-NEXT:    jirl $zero, $ra, 0
;
; CHECK64-LABEL: addRR:
; CHECK64:       # %bb.0: # %entry
; CHECK64-NEXT:    add.d $a0, $a1, $a0
; CHECK64-NEXT:    jirl $zero, $ra, 0
entry:
  %add = add nsw i32 %y, %x
  ret i32 %add
}
