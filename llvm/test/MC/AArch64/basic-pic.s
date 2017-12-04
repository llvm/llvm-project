// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o -| llvm-objdump -r - | FileCheck %s

// CHECK: RELOCATION RECORDS FOR [.rela.text]

	.file	"/home/espindola/llvm/llvm/test/CodeGen/AArch64/basic-pic.ll"
	.text
	.globl	get_globalvar
	.type	get_globalvar,@function
get_globalvar:                          // @get_globalvar
	.cfi_startproc
// %bb.0:
	adrp	x0, :got:var
	ldr	x0, [x0, #:got_lo12:var]
	ldr	 w0, [x0]
	ret
.Ltmp0:
	.size	get_globalvar, .Ltmp0-get_globalvar
	.cfi_endproc

// CHECK: R_AARCH64_ADR_GOT_PAGE var
// CHECK: R_AARCH64_LD64_GOT_LO12_NC var

	.globl	get_globalvaraddr
	.type	get_globalvaraddr,@function
get_globalvaraddr:                      // @get_globalvaraddr
	.cfi_startproc
// %bb.0:
	adrp	x0, :got:var
	ldr	x0, [x0, #:got_lo12:var]
	ret
.Ltmp1:
	.size	get_globalvaraddr, .Ltmp1-get_globalvaraddr
	.cfi_endproc
// CHECK: R_AARCH64_ADR_GOT_PAGE var
// CHECK: R_AARCH64_LD64_GOT_LO12_NC var

	.globl	get_hiddenvar
	.type	get_hiddenvar,@function
get_hiddenvar:                          // @get_hiddenvar
	.cfi_startproc
// %bb.0:
	adrp	x0, hiddenvar
	ldr	w0, [x0, #:lo12:hiddenvar]
	ret
.Ltmp2:
	.size	get_hiddenvar, .Ltmp2-get_hiddenvar
	.cfi_endproc
// CHECK: R_AARCH64_ADR_PREL_PG_HI21 hiddenvar
// CHECK: R_AARCH64_LDST32_ABS_LO12_NC hiddenvar

	.globl	get_hiddenvaraddr
	.type	get_hiddenvaraddr,@function
get_hiddenvaraddr:                      // @get_hiddenvaraddr
	.cfi_startproc
// %bb.0:
	adrp	x0, hiddenvar
	add	x0, x0, #:lo12:hiddenvar
	ret
.Ltmp3:
	.size	get_hiddenvaraddr, .Ltmp3-get_hiddenvaraddr
	.cfi_endproc
// CHECK: R_AARCH64_ADR_PREL_PG_HI21 hiddenvar
// CHECK: R_AARCH64_ADD_ABS_LO12_NC hiddenvar

	.globl	get_func
	.type	get_func,@function
get_func:                               // @get_func
	.cfi_startproc
// %bb.0:
	adrp	x0, :got:get_func
	ldr	x0, [x0, #:got_lo12:get_func]
	ret
.Ltmp4:
	.size	get_func, .Ltmp4-get_func
	.cfi_endproc

// Particularly important that the ADRP gets a relocation, LLVM tends to think
// it can relax it because it knows where get_func is. It can't!
// CHECK: R_AARCH64_ADR_GOT_PAGE get_func
// CHECK: R_AARCH64_LD64_GOT_LO12_NC get_func

	.type	var,@object             // @var
	.bss
	.globl	var
	.align	2
var:
	.word	0                       // 0x0
	.size	var, 4

	.hidden	hiddenvar               // @hiddenvar
	.type	hiddenvar,@object
	.globl	hiddenvar
	.align	2
hiddenvar:
	.word	0                       // 0x0
	.size	hiddenvar, 4


