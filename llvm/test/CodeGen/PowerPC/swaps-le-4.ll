; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O3 < %s | FileCheck %s

; This test verifies that VSX swap optimization works when an implicit
; subregister is present (in this case, in the XXPERMDI associated with
; the store).

define void @bar() {
entry:
  %x = alloca <2 x i64>, align 16
  %0 = bitcast <2 x i64>* %x to i8*
  call void @llvm.lifetime.start(i64 16, i8* %0)
  %arrayidx = getelementptr inbounds <2 x i64>, <2 x i64>* %x, i64 0, i64 0
  store <2 x i64> <i64 0, i64 1>, <2 x i64>* %x, align 16
  call void @foo(i64* %arrayidx)
  call void @llvm.lifetime.end(i64 16, i8* %0)
  ret void
}

; CHECK-LABEL: @bar
; CHECK: lxvd2x
; CHECK: stxvd2x
; CHECK-NOT: xxswapd

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @foo(i64*)
declare void @llvm.lifetime.end(i64, i8* nocapture)

