; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s


declare swifttailcc void @swifttail_callee()
define swifttailcc void @swifttail() {
; CHECK-LABEL: swifttail:
; CHECK-NOT: popq %r14
  call void asm "","~{r14}"()
  tail call swifttailcc void @swifttail_callee()
  ret void
}

define swifttailcc void @no_preserve_swiftself() {
; CHECK-LABEL: no_preserve_swiftself:
; CHECK-NOT: popq %r13
  call void asm "","~{r13}"()
  ret void
}
