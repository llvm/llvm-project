; RUN: llc -march=sparc < %s

;; getBooleanContents on Sparc used to claim that no bits mattered
;; other than the first for SELECT. Thus, the 'trunc' got eliminated
;; as redundant. But, cmp does NOT ignore the other bits!

; CHECK-LABEL: select_mask:
; CHECK: ldub [%o0], [[R:%[goli][0-7]]]
; CHECK: and [[R]], 1, [[V:%[goli][0-7]]]
; CHECK: cmp [[V]], 0
define i32 @select_mask(ptr %this) {
entry:
  %bf.load2 = load i8, ptr %this, align 4
  %bf.cast5 = trunc i8 %bf.load2 to i1
  %cond = select i1 %bf.cast5, i32 2, i32 0
  ret i32 %cond
}
