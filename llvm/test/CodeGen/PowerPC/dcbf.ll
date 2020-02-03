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

; Function Attrs: nounwind
define void @dcbfl_test(i8* %a) {
entry:
  tail call void @llvm.ppc.dcbfl(i8* %a)
; CHECK-LABEL: @dcbfl_test
; CHECK: dcbfl 0, r3
; CHECK-NEXT: blr
ret void
}

declare void @llvm.ppc.dcbfl(i8*)

; Function Attrs: nounwind
define void @dcbflp_test(i8* %a) {
entry:
  %add.a = getelementptr inbounds i8, i8* %a, i64 3
  tail call void @llvm.ppc.dcbflp(i8* %add.a)
; CHECK-LABEL: @dcbflp_test
; CHECK: addi r3, r3, 3
; CHECK-NEXT: dcbflp 0, r3
; CHECK-NEXT: blr
ret void
}

declare void @llvm.ppc.dcbflp(i8*)
