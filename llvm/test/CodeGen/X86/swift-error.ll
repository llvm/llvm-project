; RUN: llc -mtriple x86_64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

%swift.error = type opaque

declare swiftcc void @f(ptr swifterror)

define swiftcc void @g(ptr, ptr, ptr, ptr, ptr swifterror %error) {
entry:
  call swiftcc void @f(ptr nonnull nocapture swifterror %error)
  ret void
}

; CHECK-LABEL: g
; CHECK-NOT: pushq   %r12
; CHECK: callq   f
; CHECK-NOT: popq    %r12
; CHECK: retq

