; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension of an outgoing i32 call argument.

define void @caller() {
  call void @bar_Struct(i32 123)
  ret void
}

declare void @bar_Struct(i32 %Arg)

; CHECK: Narrow integer argument must have a valid extension type
