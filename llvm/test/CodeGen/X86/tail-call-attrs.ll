; RUN: llc -mtriple=x86_64-apple-darwin -o - %s | FileCheck %s

; Simple case: completely identical returns, even with extensions, shouldn't be
; a barrier to tail calls.
declare zeroext i1 @give_bool()
define zeroext i1 @test_bool() {
; CHECK-LABEL: test_bool:
; CHECK: jmp
  %call = tail call zeroext i1 @give_bool()
  ret i1 %call
}

; Here, there's more zero extension to be done between the call and the return,
; so a tail call is impossible (well, according to current Clang practice
; anyway. The AMD64 ABI isn't crystal clear on the matter).
; FIXME: The high 24 bits returned from test_i32 are undefined; do tail call!
declare zeroext i32 @give_i32()
define zeroext i8 @test_i32() {
; CHECK-LABEL: test_i32:
; CHECK: callq _give_i32
; CHECK: ret

  %call = tail call zeroext i32 @give_i32()
  %val = trunc i32 %call to i8
  ret i8 %val
}

; Here, one function is zeroext and the other is signext. To the extent that
; these both mean something they are incompatible so no tail call is possible.
; FIXME: The high 16 bits returned are undefined; do tail call!
declare zeroext i16 @give_unsigned_i16()
define signext i16 @test_incompatible_i16() {
; CHECK-LABEL: test_incompatible_i16:
; CHECK: callq _give_unsigned_i16
; CHECK: ret

  %call = tail call zeroext i16 @give_unsigned_i16()
  ret i16 %call
}

declare inreg i32 @give_i32_inreg()
define i32 @test_inreg_to_normal() {
; CHECK-LABEL: test_inreg_to_normal:
; CHECK: callq _give_i32_inreg
; CHECK: ret
  %val = tail call inreg i32 @give_i32_inreg()
  ret i32 %val
}

define inreg i32 @test_normal_to_inreg() {
; CHECK-LABEL: test_normal_to_inreg:
; CHECK: callq _give_i32
; CHECK: ret
  %val = tail call i32 @give_i32()
  ret i32 %val
}
