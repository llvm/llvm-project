; RUN: llc -mtriple=m68k -global-isel -stop-after=irtranslator < %s | FileCheck %s

; CHECK: name: f
; CHECK: RTS
define void @f() {
  ret void
}
