; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O3 < %s | FileCheck %s

; This test verifies that VSX swap optimization works when an implicit
; subregister is present (in this case, in the XXPERMDI associated with
; the store).

define void @bar() {
entry:
  %x = alloca <2 x i64>, align 16
  call void @llvm.lifetime.start.p0(i64 16, ptr %x)
  store <2 x i64> <i64 0, i64 1>, ptr %x, align 16
  call void @foo(ptr %x)
  call void @llvm.lifetime.end.p0(i64 16, ptr %x)
  ret void
}

; CHECK-LABEL: @bar
; CHECK: lxvd2x
; CHECK: stxvd2x
; CHECK-NOT: xxswapd

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @foo(ptr)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

