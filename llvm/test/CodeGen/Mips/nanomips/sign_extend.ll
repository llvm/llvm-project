; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @foo(i8 zeroext %n) {
; CHECK: seb $a0, $a0
  %conv = sext i8 %n to i32
  ret i32 %conv
}

define i32 @bar(i16 zeroext %n) {
; CHECK: seh $a0, $a0
  %conv = sext i16 %n to i32
  ret i32 %conv
}
