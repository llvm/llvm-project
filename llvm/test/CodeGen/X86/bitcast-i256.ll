; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=core-avx-i < %s | FileCheck %s

define i256 @foo(<8 x i32> %a) {
  %r = bitcast <8 x i32> %a to i256
  ret i256 %r
; CHECK: foo
; CHECK: vextractf128
; CHECK: vpextrq
; CHECK: vpextrq
; CHECK: ret
}
