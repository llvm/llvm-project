; RUN: llc -mtriple=mips -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX %s
; RUN: llc -mtriple=mipsel -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX %s
; RUN: llc -mtriple=mips -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX-INV %s
; RUN: llc -mtriple=mipsel -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX-INV %s

; RUN-TODO: llc -mtriple=mips64 -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX %s
; RUN-TODO: llc -mtriple=mips64el -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX %s
; RUN-TODO: llc -mtriple=mips64 -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX-INV,O32-FPXX-INV %s
; RUN-TODO: llc -mtriple=mips64el -mattr=+o32,+fpxx < %s | FileCheck --check-prefixes=O32-FPXX-INV,O32-FPXX-INV %s

define void @fpu_clobber() nounwind {
entry:
    call void asm "# Clobber", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f12},~{$f13},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
    ret void
}

; O32-FPXX-LABEL: fpu_clobber:
; O32-FPXX-INV-NOT:   sdc1 $f0,
; O32-FPXX-INV-NOT:   sdc1 $f1,
; O32-FPXX-INV-NOT:   sdc1 $f2,
; O32-FPXX-INV-NOT:   sdc1 $f3,
; O32-FPXX-INV-NOT:   sdc1 $f4,
; O32-FPXX-INV-NOT:   sdc1 $f5,
; O32-FPXX-INV-NOT:   sdc1 $f6,
; O32-FPXX-INV-NOT:   sdc1 $f7,
; O32-FPXX-INV-NOT:   sdc1 $f8,
; O32-FPXX-INV-NOT:   sdc1 $f9,
; O32-FPXX-INV-NOT:   sdc1 $f10,
; O32-FPXX-INV-NOT:   sdc1 $f11,
; O32-FPXX-INV-NOT:   sdc1 $f12,
; O32-FPXX-INV-NOT:   sdc1 $f13,
; O32-FPXX-INV-NOT:   sdc1 $f14,
; O32-FPXX-INV-NOT:   sdc1 $f15,
; O32-FPXX-INV-NOT:   sdc1 $f16,
; O32-FPXX-INV-NOT:   sdc1 $f17,
; O32-FPXX-INV-NOT:   sdc1 $f18,
; O32-FPXX-INV-NOT:   sdc1 $f19,
; O32-FPXX-INV-NOT:   sdc1 $f21,
; O32-FPXX-INV-NOT:   sdc1 $f23,
; O32-FPXX-INV-NOT:   sdc1 $f25,
; O32-FPXX-INV-NOT:   sdc1 $f27,
; O32-FPXX-INV-NOT:   sdc1 $f29,
; O32-FPXX-INV-NOT:   sdc1 $f31,

; O32-FPXX:           addiu $sp, $sp, -48
; O32-FPXX-DAG:       sdc1 [[F20:\$f20]], [[OFF20:[0-9]+]]($sp)
; O32-FPXX-DAG:       sdc1 [[F22:\$f22]], [[OFF22:[0-9]+]]($sp)
; O32-FPXX-DAG:       sdc1 [[F24:\$f24]], [[OFF24:[0-9]+]]($sp)
; O32-FPXX-DAG:       sdc1 [[F26:\$f26]], [[OFF26:[0-9]+]]($sp)
; O32-FPXX-DAG:       sdc1 [[F28:\$f28]], [[OFF28:[0-9]+]]($sp)
; O32-FPXX-DAG:       sdc1 [[F30:\$f30]], [[OFF30:[0-9]+]]($sp)
; O32-FPXX-DAG:       ldc1 [[F20]], [[OFF20]]($sp)
; O32-FPXX-DAG:       ldc1 [[F22]], [[OFF22]]($sp)
; O32-FPXX-DAG:       ldc1 [[F24]], [[OFF24]]($sp)
; O32-FPXX-DAG:       ldc1 [[F26]], [[OFF26]]($sp)
; O32-FPXX-DAG:       ldc1 [[F28]], [[OFF28]]($sp)
; O32-FPXX-DAG:       ldc1 [[F30]], [[OFF30]]($sp)
; O32-FPXX:           addiu $sp, $sp, 48
