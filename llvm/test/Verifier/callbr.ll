; RUN: not opt -S %s -verify 2>&1 | FileCheck %s

; CHECK: Indirect label missing from arglist.
define void @foo() {
  ; The %4 in the indirect label list is not found in the blockaddresses in the
  ; arg list (bad).
  callbr void asm sideeffect "${0:l} {1:l}", "X,X"(i8* blockaddress(@foo, %3), i8* blockaddress(@foo, %2))
  to label %1 [label %4, label %2]
1:
  ret void
2:
  ret void
3:
  ret void
4:
  ret void
}

; CHECK-NOT: Indirect label missing from arglist.
define void @bar() {
  ; %4 and %2 are both in the indirect label list and the arg list (good).
  callbr void asm sideeffect "${0:l} ${1:l}", "X,X"(i8* blockaddress(@bar, %4), i8* blockaddress(@bar, %2))
  to label %1 [label %4, label %2]
1:
  ret void
2:
  ret void
3:
  ret void
4:
  ret void
}

; CHECK-NOT: Indirect label missing from arglist.
define void @baz() {
  ; note %2 blockaddress. Such a case is possible when passing the address of
  ; a label as an input to the inline asm (both address of label and asm goto
  ; use blockaddress constants; we're testing that the indirect label list from
  ; the asm goto is in the arg list to the asm).
  callbr void asm sideeffect "${0:l} ${1:l} ${2:l}", "X,X,X"(i8* blockaddress(@baz, %4), i8* blockaddress(@baz, %2), i8* blockaddress(@baz, %3))
  to label %1 [label %3, label %4]
1:
  ret void
2:
  ret void
3:
  ret void
4:
  ret void
}
