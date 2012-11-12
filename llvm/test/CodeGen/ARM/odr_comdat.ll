; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=ARMGNUEABI

; Checking that a comdat group gets generated correctly for a static member 
; of instantiated C++ templates.
; see http://sourcery.mentor.com/public/cxx-abi/abi.html#vague-itemplate
; section 5.2.6 Instantiated templates
; "Any static member data object is emitted in a COMDAT identified by its mangled 
;  name, in any object file with a reference to its name symbol."

; Case 1: variable is not explicitly initialized, and ends up in a .bss section
; ARMGNUEABI: .section        .bss._ZN1CIiE1iE,"aGw",%nobits,_ZN1CIiE1iE,comdat
@_ZN1CIiE1iE = weak_odr global i32 0, align 4

; Case 2: variable is explicitly initialized, and ends up in a .data section
; ARMGNUEABI: .section        .data._ZN1CIiE1jE,"aGw",%progbits,_ZN1CIiE1jE,comdat
@_ZN1CIiE1jE = weak_odr global i32 12, align 4
