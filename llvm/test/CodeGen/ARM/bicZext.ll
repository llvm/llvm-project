; RUN: llc %s -o - | FileCheck %s
; ModuleID = 'bic.c'
target triple = "thumbv7-apple-ios3.0.0"

define zeroext i16 @foo16(i16 zeroext %f) nounwind readnone optsize ssp {
entry:
  ; CHECK: .thumb_func	_foo16
  ; CHECK: {{bic[^#]*#3}}
  %and = and i16 %f, -4
  ret i16 %and
}

define i32 @foo32(i32 %f) nounwind readnone optsize ssp {
entry:
  ; CHECK: .thumb_func	_foo32
  ; CHECK: {{bic[^#]*#3}}
  %and = and i32 %f, -4
  ret i32 %and
}
