; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define void @ILLBeBack() #0 {
; CHECK-LABEL: @ILLBeBack
; CHECK: bne {{[0-9]+}}, [[LABEL:\.[a-zA-Z0-9_]+]]
; CHECK: [[LABEL]]:
; CHECK: bl __xray_FunctionExit
bb:
  br i1 undef, label %bb1, label %bb8

bb1:
  %tmp = tail call i64 asm sideeffect "", "=&r,=*m,b,r,*m,~{cc}"(i64* elementtype(i64) nonnull undef, i64* nonnull undef, i64 1, i64* elementtype(i64) nonnull undef)
  %tmp2 = icmp eq i64 %tmp, 0
  br i1 %tmp2, label %bb3, label %bb8

bb3:
  %tmp4 = tail call i64 asm sideeffect "", "=&r,=*m,b,r,r,*m,~{cc}"(i64* elementtype(i64) undef, i64* undef, i64 0, i64 undef, i64* elementtype(i64) undef)
  %tmp5 = icmp eq i64 0, %tmp4
  br i1 %tmp5, label %bb6, label %bb3

bb6:
  br i1 undef, label %bb7, label %bb8

bb7:
  tail call void () undef()
  ret void

bb8:
  ret void
}

attributes #0 = { "function-instrument"="xray-always" }
