; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s


declare swifttailcc void @swifttail_callee()
define swifttailcc void @swifttail() {
; CHECK-LABEL: swifttail:
; CHECK-NOT: popq %r14
  call void asm "","~{r14}"()
  tail call swifttailcc void @swifttail_callee()
  ret void
}

define void @has_swiftasync(i8* swiftasync %in) {
; CHECK-LABEL: has_swiftasync:
; CHECK: popq %r14
  call void asm "","~{r14}"()
  ret void
}

; It's impossible to get a tail call from a function without a swiftasync
; parameter to one with unless the CC is swifttailcc. So it doesn't matter
; whether r14 is callee-saved in this case.
define void @calls_swiftasync() {
; CHECK-LABEL: calls_swiftasync:
; CHECK-NOT: jmpq _has_swiftasync
  call void asm "","~{r14}"()
  tail call void @has_swiftasync(i8* swiftasync null)
  ret void
}

define swifttailcc void @no_preserve_swiftself() {
; CHECK-LABEL: no_preserve_swiftself:
; CHECK-NOT: popq %r13
  call void asm "","~{r13}"()
  ret void
}

declare swifttailcc i8* @SwiftSelf(i8 * swiftasync %context, i8* swiftself %closure)
define swiftcc i8* @CallSwiftSelf(i8* swiftself %closure, i8* %context) {
; CHECK-LABEL: CallSwiftSelf:
; CHECK: pushq %r13
  ;call void asm "","~{r13}"() ; We get a push r13 but why not with the call
  ; below?
  %res = call swifttailcc i8* @SwiftSelf(i8 * swiftasync %context, i8* swiftself %closure)
  ret i8* %res
}
