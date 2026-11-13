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

; Check that a predicated use of $r1 doesn't end its live range when %x is
; copied to %p. The value survives if the predicate is false, so the predicated
; redefinition of $r1 must implicitly use it.
define i32 @copy(i32 %a, i32 %c) {
; CHECK-LABEL: name: copy
; CHECK:       bb.1.p:
; CHECK:         t2EORri renamable $r1, 7, 0 /* CC::eq */, killed $cpsr, def $cpsr
; CHECK-NEXT:    renamable $r1 = t2MUL renamable $r0, renamable $r0, 0 /* CC::eq */, $cpsr, implicit killed $r1
; CHECK:       bb.2.x:
entry:
  %ce = icmp eq i32 %c, 0
  br i1 %ce, label %x, label %p

p:
  call void asm sideeffect "", ""()
  %cp = icmp eq i32 %a, 1
  br i1 %cp, label %x, label %n

x:
  %e = xor i32 %c, 7
  %cx = icmp eq i32 %e, 0
  br i1 %cx, label %y, label %n

y:
  %m = mul i32 %e, %e
  %r = sub i32 %m, %e
  br label %n

n:
  %v = phi i32 [ %a, %p ], [ %e, %x ], [ %r, %y ]
  ret i32 %v
}
