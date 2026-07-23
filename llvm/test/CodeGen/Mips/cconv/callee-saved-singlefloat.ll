; RUN: llc -mtriple=mips -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64 -target-abi n32 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el -target-abi n32 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64 -target-abi n32 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,ALL-INV,N32-INV %s
; RUN: llc -mtriple=mips64el -target-abi n32 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,ALL-INV,N32-INV %s

; RUN: llc -mtriple=mips64 -target-abi n64 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el -target-abi n64 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64 -target-abi n64 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,ALL-INV,N64-INV %s
; RUN: llc -mtriple=mips64el -target-abi n64 -mattr=+single-float < %s | FileCheck --check-prefixes=ALL,ALL-INV,N64-INV %s

define void @fpu_clobber() nounwind {
entry:
        call void asm "# Clobber", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f12},~{$f13},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
        ret void
}

; ALL-LABEL: fpu_clobber:
; ALL-INV-NOT:   swc1 $f0,
; ALL-INV-NOT:   swc1 $f1,
; ALL-INV-NOT:   swc1 $f2,
; ALL-INV-NOT:   swc1 $f3,
; ALL-INV-NOT:   swc1 $f4,
; ALL-INV-NOT:   swc1 $f5,
; ALL-INV-NOT:   swc1 $f6,
; ALL-INV-NOT:   swc1 $f7,
; ALL-INV-NOT:   swc1 $f8,
; ALL-INV-NOT:   swc1 $f9,
; ALL-INV-NOT:   swc1 $f10,
; ALL-INV-NOT:   swc1 $f11,
; ALL-INV-NOT:   swc1 $f12,
; ALL-INV-NOT:   swc1 $f13,
; ALL-INV-NOT:   swc1 $f14,
; ALL-INV-NOT:   swc1 $f15,
; ALL-INV-NOT:   swc1 $f16,
; ALL-INV-NOT:   swc1 $f17,
; ALL-INV-NOT:   swc1 $f18,
; ALL-INV-NOT:   swc1 $f19,

; O32:           addiu $sp, $sp, -48
; O32-DAG:       swc1 [[F20:\$f20]], [[OFF20:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F21:\$f21]], [[OFF21:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F22:\$f22]], [[OFF22:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F23:\$f23]], [[OFF23:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F24:\$f24]], [[OFF24:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F25:\$f25]], [[OFF25:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F26:\$f26]], [[OFF26:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F27:\$f27]], [[OFF27:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F28:\$f28]], [[OFF28:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F29:\$f29]], [[OFF29:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F30:\$f30]], [[OFF30:[0-9]+]]($sp)
; O32-DAG:       swc1 [[F31:\$f31]], [[OFF31:[0-9]+]]($sp)
; O32-DAG:       lwc1 [[F20]], [[OFF20]]($sp)
; O32-DAG:       lwc1 [[F21]], [[OFF21]]($sp)
; O32-DAG:       lwc1 [[F22]], [[OFF22]]($sp)
; O32-DAG:       lwc1 [[F23]], [[OFF23]]($sp)
; O32-DAG:       lwc1 [[F24]], [[OFF24]]($sp)
; O32-DAG:       lwc1 [[F25]], [[OFF25]]($sp)
; O32-DAG:       lwc1 [[F26]], [[OFF26]]($sp)
; O32-DAG:       lwc1 [[F27]], [[OFF27]]($sp)
; O32-DAG:       lwc1 [[F28]], [[OFF28]]($sp)
; O32-DAG:       lwc1 [[F29]], [[OFF29]]($sp)
; O32-DAG:       lwc1 [[F30]], [[OFF30]]($sp)
; O32-DAG:       lwc1 [[F31]], [[OFF31]]($sp)
; O32:           addiu $sp, $sp, 48

; N32:           addiu $sp, $sp, -32
; N32-DAG:       swc1 [[F20:\$f20]], [[OFF20:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f21,
; N32-DAG:       swc1 [[F22:\$f22]], [[OFF22:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f23,
; N32-DAG:       swc1 [[F24:\$f24]], [[OFF24:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f25,
; N32-DAG:       swc1 [[F26:\$f26]], [[OFF26:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f27,
; N32-DAG:       swc1 [[F28:\$f28]], [[OFF28:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f29,
; N32-DAG:       swc1 [[F30:\$f30]], [[OFF30:[0-9]+]]($sp)
; N32-INV-NOT:   swc1 $f31,
; N32-DAG:       lwc1 [[F20]], [[OFF20]]($sp)
; N32-DAG:       lwc1 [[F22]], [[OFF22]]($sp)
; N32-DAG:       lwc1 [[F24]], [[OFF24]]($sp)
; N32-DAG:       lwc1 [[F26]], [[OFF26]]($sp)
; N32-DAG:       lwc1 [[F28]], [[OFF28]]($sp)
; N32-DAG:       lwc1 [[F30]], [[OFF30]]($sp)
; N32:           addiu $sp, $sp, 32

; N64:           addiu $sp, $sp, -32
; N64-INV-NOT:   swc1 $f20,
; N64-INV-NOT:   swc1 $f21,
; N64-INV-NOT:   swc1 $f22,
; N64-INV-NOT:   swc1 $f23,
; N64-DAG:       swc1 [[F24:\$f24]], [[OFF24:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F25:\$f25]], [[OFF25:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F26:\$f26]], [[OFF26:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F27:\$f27]], [[OFF27:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F28:\$f28]], [[OFF28:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F29:\$f29]], [[OFF29:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F30:\$f30]], [[OFF30:[0-9]+]]($sp)
; N64-DAG:       swc1 [[F31:\$f31]], [[OFF31:[0-9]+]]($sp)
; N64-DAG:       lwc1 [[F24]], [[OFF24]]($sp)
; N64-DAG:       lwc1 [[F25]], [[OFF25]]($sp)
; N64-DAG:       lwc1 [[F26]], [[OFF26]]($sp)
; N64-DAG:       lwc1 [[F27]], [[OFF27]]($sp)
; N64-DAG:       lwc1 [[F28]], [[OFF28]]($sp)
; N64-DAG:       lwc1 [[F29]], [[OFF29]]($sp)
; N64-DAG:       lwc1 [[F30]], [[OFF30]]($sp)
; N64-DAG:       lwc1 [[F31]], [[OFF31]]($sp)
; N64:           addiu $sp, $sp, 32