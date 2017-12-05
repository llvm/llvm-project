// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | llvm-readobj -r | FileCheck %s

// Check that the appropriate relocations were created.

// CHECK: Relocations [
// CHECK:   Section {{.*}} .rel.text {
// CHECK:     R_MIPS_TLS_LDM
// CHECK:     R_MIPS_TLS_DTPREL_HI16
// CHECK:     R_MIPS_TLS_DTPREL_LO16
// CHECK:   }
// CHECK: ]

	.text
	.abicalls
	.section	.mdebug.abi32,"",@progbits
	.file	"/home/espindola/llvm/llvm/test/MC/Mips/elf-tls.ll"
	.text
	.globl	f1
	.align	2
	.type	f1,@function
	.set	nomips16
	.ent	f1
f1:                                     # @f1
	.frame	$sp,24,$ra
	.mask 	0x80000000,-4
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addiu	$sp, $sp, -24
	sw	$ra, 20($sp)            # 4-byte Folded Spill
	addu	$gp, $2, $25
	lw	$25, %call16(__tls_get_addr)($gp)
	jalr	$25
	addiu	$4, $gp, %tlsgd(t1)
	lw	$2, 0($2)
	lw	$ra, 20($sp)            # 4-byte Folded Reload
	jr	$ra
	addiu	$sp, $sp, 24
	.set	at
	.set	macro
	.set	reorder
	.end	f1
$tmp0:
	.size	f1, ($tmp0)-f1

	.globl	f2
	.align	2
	.type	f2,@function
	.set	nomips16
	.ent	f2
f2:                                     # @f2
	.frame	$sp,24,$ra
	.mask 	0x80000000,-4
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addiu	$sp, $sp, -24
	sw	$ra, 20($sp)            # 4-byte Folded Spill
	addu	$gp, $2, $25
	lw	$25, %call16(__tls_get_addr)($gp)
	jalr	$25
	addiu	$4, $gp, %tlsgd(t2)
	lw	$2, 0($2)
	lw	$ra, 20($sp)            # 4-byte Folded Reload
	jr	$ra
	addiu	$sp, $sp, 24
	.set	at
	.set	macro
	.set	reorder
	.end	f2
$tmp1:
	.size	f2, ($tmp1)-f2

	.globl	f3
	.align	2
	.type	f3,@function
	.set	nomips16
	.ent	f3
f3:                                     # @f3
	.frame	$sp,24,$ra
	.mask 	0x80000000,-4
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
# %bb.0:                                # %entry
	lui	$2, %hi(_gp_disp)
	addiu	$2, $2, %lo(_gp_disp)
	addiu	$sp, $sp, -24
	sw	$ra, 20($sp)            # 4-byte Folded Spill
	addu	$gp, $2, $25
	lw	$25, %call16(__tls_get_addr)($gp)
	jalr	$25
	addiu	$4, $gp, %tlsldm(f3.i)
	lui	$1, %dtprel_hi(f3.i)
	addu	$1, $1, $2
	lw	$2, %dtprel_lo(f3.i)($1)
	addiu	$2, $2, 1
	sw	$2, %dtprel_lo(f3.i)($1)
	lw	$ra, 20($sp)            # 4-byte Folded Reload
	jr	$ra
	addiu	$sp, $sp, 24
	.set	at
	.set	macro
	.set	reorder
	.end	f3
$tmp2:
	.size	f3, ($tmp2)-f3

	.type	t1,@object              # @t1
	.section	.tbss,"awT",@nobits
	.globl	t1
	.align	2
t1:
	.4byte	0                       # 0x0
	.size	t1, 4

	.type	f3.i,@object            # @f3.i
	.section	.tdata,"awT",@progbits
	.align	2
f3.i:
	.4byte	1                       # 0x1
	.size	f3.i, 4


	.text
