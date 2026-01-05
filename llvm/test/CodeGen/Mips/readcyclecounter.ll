;RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32r2 < %s | FileCheck %s --check-prefix=MIPSEL
;RUN: llc -mtriple=mips64el-linux-gnuabi64 -mcpu=mips64r2 < %s | FileCheck %s --check-prefix=MIPS64EL
;RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips2 < %s | FileCheck %s --check-prefix=MIPSEL
;RUN: llc -mtriple=mips64el-linux-gnuabi64 -mcpu=mips3 < %s | FileCheck %s --check-prefix=MIPS64EL
;RUN: llc -mtriple=mipsel -mcpu=mips32r2 < %s | FileCheck %s --check-prefix=MIPSEL
;RUN: llc -mtriple=mips64el -mcpu=mips64r2 < %s | FileCheck %s --check-prefix=MIPS64EL
;RUN: llc -mtriple=mipsel -mcpu=mips2 < %s | FileCheck %s --check-prefix=MIPSEL_NOT_SUPPORTED
;RUN: llc -mtriple=mips64el -mcpu=mips3 < %s | FileCheck %s --check-prefix=MIPS64EL_NOT_SUPPORTED

declare i64 @llvm.readcyclecounter() nounwind readnone

define i64 @test_readcyclecounter() nounwind {
; MIPSEL-LABEL: test_readcyclecounter:
; MIPSEL:       # %bb.0: # %entry
; MIPSEL-NEXT:    .set push
; MIPSEL-NEXT:    .set mips32r2
; MIPSEL-NEXT:    rdhwr $2, $hwr_cc
; MIPSEL-NEXT:    .set pop
; MIPSEL-NEXT:    jr $ra
; MIPSEL-NEXT:    addiu $3, $zero, 0
;
; MIPSEL_NOT_SUPPORTED-LABEL: test_readcyclecounter:
; MIPSEL_NOT_SUPPORTED:       # %bb.0: # %entry
; MIPSEL_NOT_SUPPORTED-NEXT:    addiu $2, $zero, 0
; MIPSEL_NOT_SUPPORTED-NEXT:    jr $ra
; MIPSEL_NOT_SUPPORTED-NEXT:    addiu $3, $zero, 0
;
; MIPS64EL-LABEL: test_readcyclecounter:
; MIPS64EL:       # %bb.0: # %entry
; MIPS64EL-NEXT:    .set push
; MIPS64EL-NEXT:    .set mips32r2
; MIPS64EL-NEXT:    rdhwr $2, $hwr_cc
; MIPS64EL-NEXT:    .set pop
; MIPS64EL-NEXT:    jr $ra
; MIPS64EL-NEXT:    nop
;
; MIPS64EL_NOT_SUPPORTED-LABEL: test_readcyclecounter:
; MIPS64EL_NOT_SUPPORTED:       # %bb.0: # %entry
; MIPS64EL_NOT_SUPPORTED-NEXT:    jr $ra
; MIPS64EL_NOT_SUPPORTED-NEXT:    daddiu $2, $zero, 0
entry:
  %tmp0 = tail call i64 @llvm.readcyclecounter()
  ret i64 %tmp0
}

