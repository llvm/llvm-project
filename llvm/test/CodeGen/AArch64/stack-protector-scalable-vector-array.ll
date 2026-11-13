; Test that an alloca of an array of scalable vectors does not crash the stack
; protector. ContainsProtectableArray used to compare the array's alloc size
; against ssp-buffer-size through TypeSize's implicit fixed-width conversion,
; which aborts for scalable sizes.
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve < %s -o - | FileCheck %s

declare void @sink(ptr)

define void @strong() sspstrong {
; CHECK-LABEL: strong:
; CHECK: __stack_chk_guard
  %a = alloca [2 x <vscale x 4 x i32>]
  call void @sink(ptr %a)
  ret void
}

define void @req() sspreq {
; CHECK-LABEL: req:
; CHECK: __stack_chk_guard
  %a = alloca [2 x <vscale x 4 x i32>]
  call void @sink(ptr %a)
  ret void
}
