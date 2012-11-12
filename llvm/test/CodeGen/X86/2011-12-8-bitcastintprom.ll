; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 | FileCheck %s

; Make sure that the conversion between v4i8 to v2i16 is not a simple bitcast.
; CHECK: prom_bug
; CHECK: shufb
; CHECK: movd
; CHECK: movw
; CHECK: ret
define void @prom_bug(<4 x i8> %t, i16* %p) {
  %r = bitcast <4 x i8> %t to <2 x i16>
  %o = extractelement <2 x i16> %r, i32 0
  store i16 %o, i16* %p
  ret void
}

