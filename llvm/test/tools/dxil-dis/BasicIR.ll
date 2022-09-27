; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

; CHECK: define i32 @foo(i32 %X, i32 %Y) {
; CHECK:   %Z = sub i32 %X, %Y
; CHECK:   %Q = add i32 %Z, %Y
; CHECK:   ret i32 %Q
; CHECK: }

target triple = "dxil-unknown-shadermodel6.7-library"

define i32 @foo(i32 %X, i32 %Y) {
  %Z = sub i32 %X, %Y
  %Q = add i32 %Z, %Y
  ret i32 %Q
}
