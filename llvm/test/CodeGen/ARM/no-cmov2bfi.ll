; RUN: llc < %s -mtriple=thumbv7 | FileCheck --check-prefix=CHECK-NOBFI %s

declare zeroext i1 @dummy()

define i8 @test(i8 %a1, i1 %c) {
; CHECK-NOBFI-NOT: bfi
; CHECK-NOBFI: bl      dummy
; CHECK-NOBFI: cmp     r0, #0
; CHECK-NOBFI: it      ne
; CHECK-NOBFI: orrne   [[REG:r[0-9]+]], [[REG]], #8
; CHECK-NOBFI: mov     r0, [[REG]]

  %1 = and i8 %a1, -9
  %2 = select i1 %c, i8 %1, i8 %a1
  %3 = tail call zeroext i1 @dummy()
  %4 = or i8 %2, 8
  %ret = select i1 %3, i8 %4, i8 %2
  ret i8 %ret
}
