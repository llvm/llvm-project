; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mattr=+secure-plt -relocation-model=pic | FileCheck %s

; This variant of ppc32-pic-large.ll checks that a strictfp call sets
; r30 for the secure PLT.

declare void @call_foo()

define void @foo() {
entry:
  call void @call_foo() #0
  ret void
}

attributes #0 = { strictfp }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}

; CHECK:  addis 30, 30, .LTOC-.L0$pb@ha
; CHECK:  addi 30, 30, .LTOC-.L0$pb@l
; CHECK:  bl call_foo@PLT+32768
