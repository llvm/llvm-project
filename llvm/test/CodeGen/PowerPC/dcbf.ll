; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:     -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:     -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr --filetype=obj | \
; RUN:     llvm-objdump --mcpu=future -dr - | FileCheck %s --check-prefix=CHECK-O

; Function Attrs: nounwind
define void @dcbf_test(ptr %a) {
entry:
  tail call void @llvm.ppc.dcbf(ptr %a)
; CHECK-S-LABEL: @dcbf_test
; CHECK-S: dcbf 0, r3
; CHECK-S-NEXT: blr
; CHECK-O-LABEL: <dcbf_test>:
; CHECK-O:        0: ac 18 00 7c        dcbf 0, 3
; CHECK-O-NEXT:   4: 20 00 80 4e        blr
ret void
}

declare void @llvm.ppc.dcbf(ptr)

; Function Attrs: nounwind
define void @dcbfl_test(ptr %a) {
entry:
  tail call void @llvm.ppc.dcbfl(ptr %a)
; CHECK-S-LABEL: @dcbfl_test
; CHECK-S: dcbfl 0, r3
; CHECK-S-NEXT: blr
; CHECK-O-LABEL: <dcbfl_test>:
; CHECK-O:        20: ac 18 20 7c       dcbfl 0, 3
; CHECK-O-NEXT:   24: 20 00 80 4e       blr
ret void
}

declare void @llvm.ppc.dcbfl(ptr)

; Function Attrs: nounwind
define void @dcbflp_test(ptr %a) {
entry:
  %add.a = getelementptr inbounds i8, ptr %a, i64 3
  tail call void @llvm.ppc.dcbflp(ptr %add.a)
; CHECK-S-LABEL: @dcbflp_test
; CHECK-S: addi r3, r3, 3
; CHECK-S-NEXT: dcbflp 0, r3
; CHECK-S-NEXT: blr
; CHECK-O-LABEL: <dcbflp_test>:
; CHECK-O:        40: 03 00 63 38       addi 3, 3, 3
; CHECK-O-NEXT:   44: ac 18 60 7c       dcbflp 0, 3
; CHECK-O-NEXT:   48: 20 00 80 4e       blr
ret void
}

declare void @llvm.ppc.dcbflp(ptr)
