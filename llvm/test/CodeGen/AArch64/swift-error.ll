; RUN: llc -mtriple aarch64-unknown-linux-gnu -filetype asm -o - %s | FileCheck %s

%swift.error = type opaque

declare swiftcc void @f(%swift.error** swifterror)

define swiftcc void @g(i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %swift.error** swifterror %error) {
entry:
  call swiftcc void @f(%swift.error** nonnull nocapture swifterror %error)
  ret void
}

; CHEECK-LABEL: g
; CHECK: str x30, [sp, #-16]!
; CHECK: bl f
; CHECK: ldr x30, [sp], #16
; CHECK: ret

