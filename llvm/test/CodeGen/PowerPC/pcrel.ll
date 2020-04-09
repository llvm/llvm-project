; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=future -dr - | FileCheck %s --check-prefix=CHECK-O

; Constant Pool Index.
; CHECK-S-LABEL: ConstPool
; CHECK-S:       plfd f1, .LCPI0_0@PCREL(0), 1
; CHECK-S:       blr

; CHECK-O-LABEL: ConstPool
; CHECK-O:       plfd 1, 0(0), 1
; CHECK-O-NEXT:  R_PPC64_PCREL34  .rodata.cst8
; CHECK-O:       blr
define dso_local double @ConstPool() local_unnamed_addr {
  entry:
    ret double 0x406ECAB439581062
}


