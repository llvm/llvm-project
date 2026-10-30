; RUN: llc -mtriple=thumbv7-apple-ios -stop-after=if-converter -o - %s | FileCheck %s

define i32 @simple(i32 %a, i32 %c) {
; CHECK-LABEL: name: simple
; CHECK:         t2EORri renamable $r1, 7, 1 /* CC::ne */, $cpsr, $noreg
; CHECK-NEXT:    renamable $r1 = t2MUL renamable $r0, renamable $r0, 1 /* CC::ne */, $cpsr, implicit killed $r1
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %else, label %then

then:
  %x = xor i32 %c, 7
  %y = mul i32 %x, %x
  %w = sub i32 %y, %x
  ret i32 %w

else:
  %e1 = add i32 %c, 9
  %e = mul i32 %e1, %c
  ret i32 %e
}
