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

declare swifttailcc ptr @SwiftSelf(ptr swiftasync %context, ptr swiftself %closure)
define swiftcc ptr @CallSwiftSelf(ptr swiftself %closure, ptr %context) {
; CHECK-LABEL: CallSwiftSelf:
; CHECK: pushq %r13
  ;call void asm "","~{r13}"() ; We get a push r13 but why not with the call
  ; below?
  %res = call swifttailcc ptr @SwiftSelf(ptr swiftasync %context, ptr swiftself %closure)
  ret ptr %res
}
