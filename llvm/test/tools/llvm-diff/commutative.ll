; Check that commutative operations are considered equal when parameters are exchanged.
;
; Replace '%a, %b' with '%b, %a' in 'or' (commutative) and 'sub' (non-commutative) operations.
;
; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/^\(  .*\) i1 %a, %b$/\1 i1 %b, %a/' > %t.ll
; RUN: not llvm-diff --commutative %s %t.ll 2>&1 | FileCheck %s

; CHECK:      in function choice:
; CHECK-NEXT:   in block %entry:
; CHECK-NEXT:     >   %1 = sub i1 %b, %a
; CHECK-NEXT:     <   %1 = sub i1 %a, %b

define i1 @choice(i1 %a, i1 %b) {
entry:
  %0 = or i1 %a, %b
  %1 = sub i1 %a, %b
  ret i1 %0
}
