; RUN: llc -mtriple=arm64-apple-ios7.0 -verify-machineinstrs < %s | FileCheck %s

; CallingConv.td requires a bitcast for vector arguments. Make sure we're
; actually capable of that (the test was omitted from LowerFormalArguments).

define void @test_bitcast_lower(<2 x i32> %a) {
; CHECK-LABEL: test_bitcast_lower:

  ret void
; CHECK: ret
}
