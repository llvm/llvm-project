; RUN: llc -mtriple=m68k -global-isel -stop-after=irtranslator < %s | FileCheck %s

; CHECK: name: noArgRetVoid
; CHECK: RTS
define void @noArgRetVoid() {
  ret void
}
