; RUN: llc < %s -mtriple=x86_64-apple-macosx10.7 -disable-cfi | FileCheck --check-prefix=CHECK-64-D11 %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.6 -disable-cfi | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.5 -disable-cfi | FileCheck --check-prefix=CHECK-64-D89 %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.6 -disable-cfi | FileCheck --check-prefix=CHECK-I686-D10 %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.5 -disable-cfi | FileCheck --check-prefix=CHECK-I686-D89 %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.4 -disable-cfi | FileCheck --check-prefix=CHECK-I686-D89 %s

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


; CHECK-64-D11: Ltmp13:
; CHECK-64-D11-NEXT: Ltmp14 = L_foo-Ltmp13                   ## FDE initial location
; CHECK-64-D11-NEXT:         .quad   Ltmp14

; CHECK-64-D11: Ltmp20:
; CHECK-64-D11-NEXT: Ltmp21 = Ltmp2-Ltmp20                   ## FDE initial location
; CHECK-64-D11-NEXT:         .quad   Ltmp21


; CHECK-64-D89: Ltmp12:
; CHECK-64-D89-NEXT: .quad	L_foo-Ltmp12                   ## FDE initial location
; CHECK-64-D89-NEXT: Ltmp13 = (Ltmp0-L_foo)-0                   ## FDE address range
; CHECK-64-D89-NEXT:         .quad   Ltmp13

; CHECK-64-D89: Ltmp18:
; CHECK-64-D89-NEXT: .quad	Ltmp2-Ltmp18                   ## FDE initial location
; CHECK-64-D89-NEXT: Ltmp19 = (Ltmp4-Ltmp2)-0                   ## FDE address range
; CHECK-64-D89-NEXT:         .quad   Ltmp19


; CHECK-I686-D10: Ltmp12:
; CHECK-I686-D10-NEXT: Ltmp13 = L_foo-Ltmp12                   ## FDE initial location
; CHECK-I686-D10-NEXT:         .long   Ltmp13

; CHECK-I686-D10: Ltmp19:
; CHECK-I686-D10-NEXT: Ltmp20 = Ltmp2-Ltmp19                   ## FDE initial location
; CHECK-I686-D10-NEXT:         .long   Ltmp20


; CHECK-I686-D89: Ltmp12:
; CHECK-I686-D89-NEXT: .long	L_foo-Ltmp12                   ## FDE initial location
; CHECK-I686-D89-NEXT: Ltmp13 = (Ltmp0-L_foo)-0                   ## FDE address range
; CHECK-I686-D89-NEXT:         .long   Ltmp13

; CHECK-I686-D89: Ltmp18:
; CHECK-I686-D89-NEXT: .long	Ltmp2-Ltmp18                   ## FDE initial location
; CHECK-I686-D89-NEXT: Ltmp19 = (Ltmp4-Ltmp2)-0                   ## FDE address range
; CHECK-I686-D89-NEXT:         .long   Ltmp19

