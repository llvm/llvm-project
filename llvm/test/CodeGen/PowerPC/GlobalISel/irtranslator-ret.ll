; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -global-isel -verify-machineinstrs -stop-after=irtranslator < %s | FileCheck %s

; CHECK: name: f
; CHECK: BLR8
define void @f() {
  ret void
}
