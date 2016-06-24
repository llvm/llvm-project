; RUN: llc -march=mips < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -march=mipsel < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -march=mips < %s | FileCheck --check-prefixes=ALL,O32-INV %s
; RUN: llc -march=mipsel < %s | FileCheck --check-prefixes=ALL,O32-INV %s

; RUN-TODO: llc -march=mips64 -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -march=mips64el -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -march=mips64 -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32-INV %s
; RUN-TODO: llc -march=mips64el -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32-INV %s

; RUN: llc -march=mips64 -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -march=mips64el -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -march=mips64 -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32-INV %s
; RUN: llc -march=mips64el -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32-INV %s

; RUN: llc -march=mips64 -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -march=mips64el -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -march=mips64 -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64-INV %s
; RUN: llc -march=mips64el -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64-INV %s

; Test the callee-saved registers are callee-saved as specified by section
; 2 of the MIPSpro N32 Handbook and section 3 of the SYSV ABI spec.

define void @gpr_clobber() nounwind {
entry:
        ; Clobbering the stack pointer is a bad idea so we'll skip that one
        call void asm "# Clobber", "~{$0},~{$1},~{$2},~{$3},~{$4},~{$5},~{$6},~{$7},~{$8},~{$9},~{$10},~{$11},~{$12},~{$13},~{$14},~{$15},~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$22},~{$23},~{$24},~{$25},~{$26},~{$27},~{$28},~{$30},~{$31}"()
        ret void
}

; ALL-LABEL: gpr_clobber:
; O32:           addiu $sp, $sp, -40
; O32-INV-NOT:   sw $0,
; O32-INV-NOT:   sw $1,
; O32-INV-NOT:   sw $2,
; O32-INV-NOT:   sw $3,
; O32-INV-NOT:   sw $4,
; O32-INV-NOT:   sw $5,
; O32-INV-NOT:   sw $6,
; O32-INV-NOT:   sw $7,
; O32-INV-NOT:   sw $8,
; O32-INV-NOT:   sw $9,
; O32-INV-NOT:   sw $10,
; O32-INV-NOT:   sw $11,
; O32-INV-NOT:   sw $12,
; O32-INV-NOT:   sw $13,
; O32-INV-NOT:   sw $14,
; O32-INV-NOT:   sw $15,
; O32-DAG:       sw [[G16:\$16]], [[OFF16:[0-9]+]]($sp)
; O32-DAG:       sw [[G17:\$17]], [[OFF17:[0-9]+]]($sp)
; O32-DAG:       sw [[G18:\$18]], [[OFF18:[0-9]+]]($sp)
; O32-DAG:       sw [[G19:\$19]], [[OFF19:[0-9]+]]($sp)
; O32-DAG:       sw [[G20:\$20]], [[OFF20:[0-9]+]]($sp)
; O32-DAG:       sw [[G21:\$21]], [[OFF21:[0-9]+]]($sp)
; O32-DAG:       sw [[G22:\$22]], [[OFF22:[0-9]+]]($sp)
; O32-DAG:       sw [[G23:\$23]], [[OFF23:[0-9]+]]($sp)
; O32-INV-NOT:   sw $24,
; O32-INV-NOT:   sw $25,
; O32-INV-NOT:   sw $26,
; O32-INV-NOT:   sw $27,
; O32-INV-NOT:   sw $28,
; O32-INV-NOT:   sw $29,
; O32-DAG:       sw [[G30:\$fp]], [[OFF30:[0-9]+]]($sp)
; O32-DAG:       sw [[G31:\$fp]], [[OFF31:[0-9]+]]($sp)
; O32-DAG:       lw [[G16]], [[OFF16]]($sp)
; O32-DAG:       lw [[G17]], [[OFF17]]($sp)
; O32-DAG:       lw [[G18]], [[OFF18]]($sp)
; O32-DAG:       lw [[G19]], [[OFF19]]($sp)
; O32-DAG:       lw [[G20]], [[OFF20]]($sp)
; O32-DAG:       lw [[G21]], [[OFF21]]($sp)
; O32-DAG:       lw [[G22]], [[OFF22]]($sp)
; O32-DAG:       lw [[G23]], [[OFF23]]($sp)
; O32-DAG:       lw [[G30]], [[OFF30]]($sp)
; O32-DAG:       lw [[G31]], [[OFF31]]($sp)
; O32:           addiu $sp, $sp, 40

; N32:           addiu $sp, $sp, -96
; N32-INV-NOT:   sd $0,
; N32-INV-NOT:   sd $1,
; N32-INV-NOT:   sd $2,
; N32-INV-NOT:   sd $3,
; N32-INV-NOT:   sd $4,
; N32-INV-NOT:   sd $5,
; N32-INV-NOT:   sd $6,
; N32-INV-NOT:   sd $7,
; N32-INV-NOT:   sd $8,
; N32-INV-NOT:   sd $9,
; N32-INV-NOT:   sd $10,
; N32-INV-NOT:   sd $11,
; N32-INV-NOT:   sd $12,
; N32-INV-NOT:   sd $13,
; N32-INV-NOT:   sd $14,
; N32-INV-NOT:   sd $15,
; N32-DAG:       sd [[G16:\$16]], [[OFF16:[0-9]+]]($sp)
; N32-DAG:       sd [[G17:\$17]], [[OFF17:[0-9]+]]($sp)
; N32-DAG:       sd [[G18:\$18]], [[OFF18:[0-9]+]]($sp)
; N32-DAG:       sd [[G19:\$19]], [[OFF19:[0-9]+]]($sp)
; N32-DAG:       sd [[G20:\$20]], [[OFF20:[0-9]+]]($sp)
; N32-DAG:       sd [[G21:\$21]], [[OFF21:[0-9]+]]($sp)
; N32-DAG:       sd [[G22:\$22]], [[OFF22:[0-9]+]]($sp)
; N32-DAG:       sd [[G23:\$23]], [[OFF23:[0-9]+]]($sp)
; N32-INV-NOT:   sd $24,
; N32-INV-NOT:   sd $25,
; N32-INV-NOT:   sd $26,
; N32-INV-NOT:   sd $27,
; N32-DAG:       sd [[G28:\$gp]], [[OFF28:[0-9]+]]($sp)
; N32-INV-NOT:   sd $29,
; N32-DAG:       sd [[G30:\$fp]], [[OFF30:[0-9]+]]($sp)
; N32-DAG:       sd [[G31:\$fp]], [[OFF31:[0-9]+]]($sp)
; N32-DAG:       ld [[G16]], [[OFF16]]($sp)
; N32-DAG:       ld [[G17]], [[OFF17]]($sp)
; N32-DAG:       ld [[G18]], [[OFF18]]($sp)
; N32-DAG:       ld [[G19]], [[OFF19]]($sp)
; N32-DAG:       ld [[G20]], [[OFF20]]($sp)
; N32-DAG:       ld [[G21]], [[OFF21]]($sp)
; N32-DAG:       ld [[G22]], [[OFF22]]($sp)
; N32-DAG:       ld [[G23]], [[OFF23]]($sp)
; N32-DAG:       ld [[G28]], [[OFF28]]($sp)
; N32-DAG:       ld [[G30]], [[OFF30]]($sp)
; N32-DAG:       ld [[G31]], [[OFF31]]($sp)
; N32:           addiu $sp, $sp, 96

; N64:           daddiu $sp, $sp, -96
; N64-INV-NOT:   sd $0,
; N64-INV-NOT:   sd $1,
; N64-INV-NOT:   sd $2,
; N64-INV-NOT:   sd $3,
; N64-INV-NOT:   sd $4,
; N64-INV-NOT:   sd $5,
; N64-INV-NOT:   sd $6,
; N64-INV-NOT:   sd $7,
; N64-INV-NOT:   sd $8,
; N64-INV-NOT:   sd $9,
; N64-INV-NOT:   sd $10,
; N64-INV-NOT:   sd $11,
; N64-INV-NOT:   sd $12,
; N64-INV-NOT:   sd $13,
; N64-INV-NOT:   sd $14,
; N64-INV-NOT:   sd $15,
; N64-DAG:       sd [[G16:\$16]], [[OFF16:[0-9]+]]($sp)
; N64-DAG:       sd [[G17:\$17]], [[OFF17:[0-9]+]]($sp)
; N64-DAG:       sd [[G18:\$18]], [[OFF18:[0-9]+]]($sp)
; N64-DAG:       sd [[G19:\$19]], [[OFF19:[0-9]+]]($sp)
; N64-DAG:       sd [[G20:\$20]], [[OFF20:[0-9]+]]($sp)
; N64-DAG:       sd [[G21:\$21]], [[OFF21:[0-9]+]]($sp)
; N64-DAG:       sd [[G22:\$22]], [[OFF22:[0-9]+]]($sp)
; N64-DAG:       sd [[G23:\$23]], [[OFF23:[0-9]+]]($sp)
; N64-DAG:       sd [[G30:\$fp]], [[OFF30:[0-9]+]]($sp)
; N64-DAG:       sd [[G31:\$fp]], [[OFF31:[0-9]+]]($sp)
; N64-INV-NOT:   sd $24,
; N64-INV-NOT:   sd $25,
; N64-INV-NOT:   sd $26,
; N64-INV-NOT:   sd $27,
; N64-DAG:       sd [[G28:\$gp]], [[OFF28:[0-9]+]]($sp)
; N64-INV-NOT:   sd $29,
; N64-DAG:       ld [[G16]], [[OFF16]]($sp)
; N64-DAG:       ld [[G17]], [[OFF17]]($sp)
; N64-DAG:       ld [[G18]], [[OFF18]]($sp)
; N64-DAG:       ld [[G19]], [[OFF19]]($sp)
; N64-DAG:       ld [[G20]], [[OFF20]]($sp)
; N64-DAG:       ld [[G21]], [[OFF21]]($sp)
; N64-DAG:       ld [[G22]], [[OFF22]]($sp)
; N64-DAG:       ld [[G23]], [[OFF23]]($sp)
; N64-DAG:       ld [[G28]], [[OFF28]]($sp)
; N64-DAG:       ld [[G30]], [[OFF30]]($sp)
; N64-DAG:       ld [[G31]], [[OFF31]]($sp)
; N64:           daddiu $sp, $sp, 96
