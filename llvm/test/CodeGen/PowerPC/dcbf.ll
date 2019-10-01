; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:     -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr | FileCheck %s

; Function Attrs: nounwind
define void @dcbf_test(i8* %a) {
entry:
  tail call void @llvm.ppc.dcbf(i8* %a)
; CHECK-LABEL: @dcbf_test
; CHECK: dcbf 0, r3
; CHECK-NEXT: blr
ret void
}

declare void @llvm.ppc.dcbf(i8*)
