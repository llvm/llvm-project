; RUN: llc -mtriple x86_64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

%swift.error = type opaque

declare swiftcc void @f(%swift.error** swifterror)

define swiftcc void @g(i8*, i8*, i8*, i8*, %swift.error** swifterror %error) {
entry:
  call swiftcc void @f(%swift.error** nonnull nocapture swifterror %error)
  ret void
}

; CHECK-LABEL: g
; CHECK-NOT: pushq   %r12
; CHECK: callq   f
; CHECK-NOT: popq    %r12
; CHECK: retq

