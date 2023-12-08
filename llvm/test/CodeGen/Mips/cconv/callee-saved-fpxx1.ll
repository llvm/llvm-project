; RUN: llc -march=mips -mattr=+o32,+fp64,+mips32r2 < %s \
; RUN:   | FileCheck --check-prefix=O32-FP64-INV %s
; RUN: llc -march=mipsel -mattr=+o32,+fp64,+mips32r2 < %s \
; RUN:   | FileCheck --check-prefix=O32-FP64-INV %s

; RUN: llc -march=mips -mattr=+o32,+fpxx < %s | FileCheck --check-prefix=O32-FPXX %s
; RUN: llc -march=mipsel -mattr=+o32,+fpxx < %s | FileCheck --check-prefix=O32-FPXX %s

; RUN-TODO: llc -march=mips64 -mattr=+o32,+fpxx < %s | FileCheck --check-prefix=O32-FPXX %s
; RUN-TODO: llc -march=mips64el -mattr=+o32,+fpxx < %s | FileCheck --check-prefix=O32-FPXX %s

define void @fpu_clobber() nounwind {
entry:
    call void asm "# Clobber", "~{$f21}"()
    ret void
}

; O32-FPXX-LABEL: fpu_clobber:

; O32-FPXX:           addiu $sp, $sp, -8

; O32-FP64-INV-NOT:   sdc1 $f20,
; O32-FPXX-DAG:       sdc1 [[F20:\$f20]], [[OFF20:[0-9]+]]($sp)
; O32-FPXX-DAG:       ldc1 [[F20]], [[OFF20]]($sp)

; O32-FPXX:           addiu $sp, $sp, 8
