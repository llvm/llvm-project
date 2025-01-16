; Check that commutative operations are considered equal after normalization when parameters are exchanged.
;
; Replace '%a, %b' with '%b, %a' in 'or' (commutative) and 'sub' (non-commutative) operations.
;
; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/^\(  .*\) i1 %a, %b$/\1 i1 %b, %a/' > %t.ll
; RUN: not llvm-diff                        %s %t.ll 2>&1 | FileCheck %s --check-prefix DEFAULT
; RUN: not llvm-diff --enable-ir-normalizer %s %t.ll 2>&1 | FileCheck %s --check-prefix NORMALIZE

; DEFAULT:      in function choice:
; DEFAULT-NEXT:   in block %entry:
; DEFAULT-NEXT:     >   %0 = or i1 %b, %a
; DEFAULT-NEXT:     >   %1 = sub i1 %b, %a
; DEFAULT-NEXT:     >   ret i1 %0
; DEFAULT-NEXT:     <   %0 = or i1 %a, %b
; DEFAULT-NEXT:     <   %1 = sub i1 %a, %b
; DEFAULT-NEXT:     <   ret i1 %0

; NORMALIZE:      in function choice:
; NORMALIZE-NEXT:   in block [[BB:%bb[0-9]+]]:
; NORMALIZE-NEXT:     >   %0 = sub i1 %a1, %a0
; NORMALIZE-NEXT:     <   %0 = sub i1 %a0, %a1

define i1 @choice(i1 %a, i1 %b) {
entry:
  %0 = or i1 %a, %b
  %1 = sub i1 %a, %b
  ret i1 %0
}
