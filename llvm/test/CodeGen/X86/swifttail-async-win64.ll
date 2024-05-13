; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s

declare swifttailcc void @callee()

define swifttailcc void @swift_tail() {
  call void asm "","~{r14}"()
  tail call swifttailcc void @callee()
  ret void
}

; CHECK-LABEL: swift_tail:
; CHECK-NOT: popq %r14

define void @has_swift_async(ptr swiftasync %contet) {
  call void asm "","~{r14}"()
  ret void
}

; CHECK-LABEL: has_swift_async:
; CHECK: popq    %r14

; It's impossible to get a tail call from a function without a swiftasync
; parameter to one with unless the CC is swifttailcc. So it doesn't matter
; whether r14 is callee-saved in this case.
define void @calls_swift_async() {
  call void asm "","~{r14}"()
  tail call void @has_swift_async(ptr swiftasync null)
  ret void
}

; CHECK-LABEL: calls_swift_async:
; CHECK-NOT: jmpq has_swift_async

define swifttailcc void @no_preserve_swiftself() {
  call void asm "","~{r13}"()
  ret void
}

; CHECK-LABEL: no_preserve_swiftself:
; CHECK-NOT: popq %r13

declare swifttailcc ptr @swift_self(ptr swiftasync %context, ptr swiftself %self)

define swiftcc ptr @call_swift_self(ptr swiftself %self, ptr %context) {
  ; call void asm "","~{r13}"()
  ; We get a push r13 but why not with the call below?
  %res = call swifttailcc ptr @swift_self(ptr swiftasync %context, ptr swiftself %self)
  ret ptr %res
}

; CHECK-LABEL: call_swift_self:
; CHECK: pushq %r13
