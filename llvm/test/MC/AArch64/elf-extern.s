// RUN: llvm-mc < %s -triple=aarch64-none-linux-gnu -filetype=obj | llvm-readobj -r - | FileCheck %s

// External symbols are a different concept to global variables but should still
// get relocations and so on when used.

	.file	"<stdin>"
	.text
	.globl	check_extern
	.type	check_extern,@function
check_extern:                           // @check_extern
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #16
.Ltmp2:
	.cfi_def_cfa sp, 16
	str	x30, [sp, #8]           // 8-byte Folded Spill
.Ltmp3:
	.cfi_offset x30, -8
	bl	memcpy
	mov	 x0, xzr
	ldr	x30, [sp, #8]           // 8-byte Folded Reload
	add	sp, sp, #16
	ret
.Ltmp4:
	.size	check_extern, .Ltmp4-check_extern
	.cfi_endproc


// CHECK: Relocations [
// CHECK:   Section {{.*}} .rela.text {
// CHECK:     0x{{[0-9,A-F]+}} R_AARCH64_CALL26 memcpy
// CHECK:   }
// CHECK: ]
