; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

;; Test integers selection after integers comparison

define i32 @select_eq(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_eq:
; LA32:       # %bb.0:
; LA32-NEXT:    xor $a0, $a0, $a1
; LA32-NEXT:    sltui $a0, $a0, 1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_eq:
; LA64:       # %bb.0:
; LA64-NEXT:    xor $a0, $a0, $a1
; LA64-NEXT:    sltui $a0, $a0, 1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp eq i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_ne(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_ne:
; LA32:       # %bb.0:
; LA32-NEXT:    xor $a0, $a0, $a1
; LA32-NEXT:    sltu $a0, $zero, $a0
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_ne:
; LA64:       # %bb.0:
; LA64-NEXT:    xor $a0, $a0, $a1
; LA64-NEXT:    sltu $a0, $zero, $a0
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp ne i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_ugt(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_ugt:
; LA32:       # %bb.0:
; LA32-NEXT:    sltu $a0, $a1, $a0
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_ugt:
; LA64:       # %bb.0:
; LA64-NEXT:    sltu $a0, $a1, $a0
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp ugt i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_uge(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_uge:
; LA32:       # %bb.0:
; LA32-NEXT:    sltu $a0, $a0, $a1
; LA32-NEXT:    xori $a0, $a0, 1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_uge:
; LA64:       # %bb.0:
; LA64-NEXT:    sltu $a0, $a0, $a1
; LA64-NEXT:    xori $a0, $a0, 1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp uge i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_ult(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_ult:
; LA32:       # %bb.0:
; LA32-NEXT:    sltu $a0, $a0, $a1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_ult:
; LA64:       # %bb.0:
; LA64-NEXT:    sltu $a0, $a0, $a1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp ult i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_ule(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_ule:
; LA32:       # %bb.0:
; LA32-NEXT:    sltu $a0, $a1, $a0
; LA32-NEXT:    xori $a0, $a0, 1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_ule:
; LA64:       # %bb.0:
; LA64-NEXT:    sltu $a0, $a1, $a0
; LA64-NEXT:    xori $a0, $a0, 1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp ule i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_sgt(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_sgt:
; LA32:       # %bb.0:
; LA32-NEXT:    slt $a0, $a1, $a0
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_sgt:
; LA64:       # %bb.0:
; LA64-NEXT:    slt $a0, $a1, $a0
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp sgt i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_sge(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_sge:
; LA32:       # %bb.0:
; LA32-NEXT:    slt $a0, $a0, $a1
; LA32-NEXT:    xori $a0, $a0, 1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_sge:
; LA64:       # %bb.0:
; LA64-NEXT:    slt $a0, $a0, $a1
; LA64-NEXT:    xori $a0, $a0, 1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp sge i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_slt(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_slt:
; LA32:       # %bb.0:
; LA32-NEXT:    slt $a0, $a0, $a1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_slt:
; LA64:       # %bb.0:
; LA64-NEXT:    slt $a0, $a0, $a1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp slt i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}

define i32 @select_sle(i32 signext %a, i32 signext %b, i32 %x, i32 %y) {
; LA32-LABEL: select_sle:
; LA32:       # %bb.0:
; LA32-NEXT:    slt $a0, $a1, $a0
; LA32-NEXT:    xori $a0, $a0, 1
; LA32-NEXT:    masknez $a1, $a3, $a0
; LA32-NEXT:    maskeqz $a0, $a2, $a0
; LA32-NEXT:    or $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: select_sle:
; LA64:       # %bb.0:
; LA64-NEXT:    slt $a0, $a1, $a0
; LA64-NEXT:    xori $a0, $a0, 1
; LA64-NEXT:    masknez $a1, $a3, $a0
; LA64-NEXT:    maskeqz $a0, $a2, $a0
; LA64-NEXT:    or $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %cond = icmp sle i32 %a, %b
  %res = select i1 %cond, i32 %x, i32 %y
  ret i32 %res
}
