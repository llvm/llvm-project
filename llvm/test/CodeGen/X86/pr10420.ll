; RUN: llc < %s -mtriple=x86_64-apple-macosx -disable-cfi | FileCheck %s

define private void @foo() {
       ret void
}

define void @bar() {
       call void @foo()
       ret void;
}

; CHECK: _bar:                                   ## @bar
; CHECK-NEXT: Ltmp2:

; CHECK: Ltmp12:
; CHECK-NEXT: Ltmp13 = L_foo-Ltmp12                   ## FDE initial location
; CHECK-NEXT:         .quad   Ltmp13

; CHECK: Ltmp19:
; CHECK-NEXT: Ltmp20 = Ltmp2-Ltmp19                   ## FDE initial location
; CHECK-NEXT:         .quad   Ltmp20
