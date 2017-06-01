; RUN: llvm-link -S %s %p/Inputs/comdat13.ll -o - | FileCheck %s

; In Inputs/comdat13.ll a function not in the $foo comdat (zed) references an
; internal function in the comdat $foo.
; The IR would be ilegal on ELF ("relocation refers to discarded section"),
; but COFF linkers seem to just duplicate the comdat.

$foo = comdat any
@foo = internal global i8 0, comdat
define i8* @bar() {
       ret i8* @foo
}

; CHECK: $foo = comdat any

; CHECK: @foo = internal global i8 0, comdat
; CHECK: @foo.1 = internal global i8 1, comdat($foo)

; CHECK:      define i8* @bar() {
; CHECK-NEXT:   ret i8* @foo
; CHECK-NEXT: }

; CHECK:      define i8* @zed() {
; CHECK-NEXT:   call void @bax()
; CHECK-NEXT:   ret i8* @foo.1
; CHECK-NEXT: }

; CHECK:      define internal void @bax() comdat($foo) {
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
