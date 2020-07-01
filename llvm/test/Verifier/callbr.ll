; RUN: not opt -S %s -verify 2>&1 | FileCheck %s

; CHECK: Indirect label missing from arglist.
; CHECK-NEXT: #test1
define void @test1() {
  ; The %4 in the indirect label list is not found in the blockaddresses in the
  ; arg list (bad).
  callbr void asm sideeffect "#test1", "X,X"(i8* blockaddress(@test1, %3), i8* blockaddress(@test1, %2))
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
define void @test2() {
  ; %4 and %2 are both in the indirect label list and the arg list (good).
  callbr void asm sideeffect "${0:l} ${1:l}", "X,X"(i8* blockaddress(@test2, %4), i8* blockaddress(@test2, %2))
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
define void @test3() {
  ; note %2 blockaddress. Such a case is possible when passing the address of
  ; a label as an input to the inline asm (both address of label and asm goto
  ; use blockaddress constants; we're testing that the indirect label list from
  ; the asm goto is in the arg list to the asm).
  callbr void asm sideeffect "${0:l} ${1:l} ${2:l}", "X,X,X"(i8* blockaddress(@test3, %4), i8* blockaddress(@test3, %2), i8* blockaddress(@test3, %3))
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

;; Ensure you cannot use the return value of a callbr in indirect targets.
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT: #test4
define i32 @test4(i1 %var) {
entry:
  %ret = callbr i32 asm sideeffect "#test4", "=r,X"(i8* blockaddress(@test4, %abnormal)) to label %normal [label %abnormal]

normal:
  ret i32 0

abnormal:
  ret i32 %ret
}

;; Ensure you cannot specify the same label as both normal and indirect targets.
; CHECK: Duplicate callbr destination!
; CHECK-NEXT: #test5
define i32 @test5() {
entry:
  %ret = callbr i32 asm sideeffect "#test5", "=r,X"(i8* blockaddress(@test5, %both)) to label %both [label %both]

both:
  ret i32 0
}
