; RUN: llc -mtriple=bpfel -global-isel -verify-machineinstrs -stop-after=irtranslator < %s | FileCheck %s

; CHECK: name: f
; CHECK: RET
define void @f() {
  ret void
}
