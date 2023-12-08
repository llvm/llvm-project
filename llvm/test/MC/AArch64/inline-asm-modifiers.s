// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj -mattr=+fp-armv8 < %s | llvm-objdump -r - | FileCheck %s

	.file	"<stdin>"
	.text
	.globl	test_inline_modifier_L
	.type	test_inline_modifier_L,@function
test_inline_modifier_L:                 // @test_inline_modifier_L
// %bb.0:
	//APP
	add x0, x0, #:lo12:var_simple
	//NO_APP
	//APP
	ldr x0, [x0, #:got_lo12:var_got]
	//NO_APP
	//APP
	add x0, x0, #:tlsdesc_lo12:var_tlsgd
	//NO_APP
	//APP
	add x0, x0, #:dtprel_lo12:var_tlsld
	//NO_APP
	//APP
	ldr x0, [x0, #:gottprel_lo12:var_tlsie]
	//NO_APP
	//APP
	add x0, x0, #:tprel_lo12:var_tlsle
	//NO_APP
	ret
.Ltmp0:
	.size	test_inline_modifier_L, .Ltmp0-test_inline_modifier_L

// CHECK: R_AARCH64_ADD_ABS_LO12_NC var_simple
// CHECK: R_AARCH64_LD64_GOT_LO12_NC var_got
// CHECK: R_AARCH64_TLSDESC_ADD_LO12 var_tlsgd
// CHECK: R_AARCH64_TLSLD_ADD_DTPREL_LO12 var_tlsld
// CHECK: R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC var_tlsie
// CHECK: R_AARCH64_TLSLE_ADD_TPREL_LO12 var_tlsle

	.globl	test_inline_modifier_G
	.type	test_inline_modifier_G,@function
test_inline_modifier_G:                 // @test_inline_modifier_G
// %bb.0:
	//APP
	add x0, x0, #:dtprel_hi12:var_tlsld, lsl #12
	//NO_APP
	//APP
	add x0, x0, #:tprel_hi12:var_tlsle, lsl #12
	//NO_APP
	ret
.Ltmp1:
	.size	test_inline_modifier_G, .Ltmp1-test_inline_modifier_G

// CHECK: R_AARCH64_TLSLD_ADD_DTPREL_HI12 var_tlsld
// CHECK: R_AARCH64_TLSLE_ADD_TPREL_HI12 var_tlsle

	.globl	test_inline_modifier_A
	.type	test_inline_modifier_A,@function
test_inline_modifier_A:                 // @test_inline_modifier_A
// %bb.0:
	//APP
	adrp x0, var_simple
	//NO_APP
	//APP
	adrp x0, :got:var_got
	//NO_APP
	//APP
	adrp x0, :tlsdesc:var_tlsgd
	//NO_APP
	//APP
	adrp x0, :gottprel:var_tlsie
	//NO_APP
	//APP
	ldr x0, :got:var_got
	//NO_APP
	ret
.Ltmp2:
	.size	test_inline_modifier_A, .Ltmp2-test_inline_modifier_A
// CHECK: R_AARCH64_ADR_PREL_PG_HI21 var_simple
// CHECK: R_AARCH64_ADR_GOT_PAGE var_got
// CHECK: R_AARCH64_TLSDESC_ADR_PAGE21 var_tlsgd
// CHECK: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 var_tlsie
// CHECK: R_AARCH64_GOT_LD_PREL19 var_got

	.globl	test_inline_modifier_wx
	.type	test_inline_modifier_wx,@function
test_inline_modifier_wx:                // @test_inline_modifier_wx
// %bb.0:
	mov	 w2, w0
	//APP
	add w2, w2, w2
	//NO_APP
	mov	 w2, w0
	//APP
	add w2, w2, w2
	//NO_APP
	//APP
	add x0, x0, x0
	//NO_APP
	mov	 x0, x1
	//APP
	add x0, x0, x0
	//NO_APP
	mov	 x0, x1
	//APP
	add w0, w0, w0
	//NO_APP
	//APP
	add x1, x1, x1
	//NO_APP
	//APP
	add w0, wzr, wzr
	//NO_APP
	//APP
	add x0, xzr, xzr
	//NO_APP
	ret
.Ltmp3:
	.size	test_inline_modifier_wx, .Ltmp3-test_inline_modifier_wx

	.globl	test_inline_modifier_bhsdq
	.type	test_inline_modifier_bhsdq,@function
test_inline_modifier_bhsdq:             // @test_inline_modifier_bhsdq
// %bb.0:
	//APP
	ldr b0, [sp]
	//NO_APP
	//APP
	ldr h0, [sp]
	//NO_APP
	//APP
	ldr s0, [sp]
	//NO_APP
	//APP
	ldr d0, [sp]
	//NO_APP
	//APP
	ldr q0, [sp]
	//NO_APP
	//APP
	ldr b0, [sp]
	//NO_APP
	//APP
	ldr h0, [sp]
	//NO_APP
	//APP
	ldr s0, [sp]
	//NO_APP
	//APP
	ldr d0, [sp]
	//NO_APP
	//APP
	ldr q0, [sp]
	//NO_APP
	ret
.Ltmp4:
	.size	test_inline_modifier_bhsdq, .Ltmp4-test_inline_modifier_bhsdq

	.globl	test_inline_modifier_c
	.type	test_inline_modifier_c,@function
test_inline_modifier_c:                 // @test_inline_modifier_c
// %bb.0:
	//APP
	adr x0, 3
	//NO_APP
	ret
.Ltmp5:
	.size	test_inline_modifier_c, .Ltmp5-test_inline_modifier_c

	.hidden	var_simple              // @var_simple
	.type	var_simple,@object
	.bss
	.globl	var_simple
	.align	2
var_simple:
	.word	0                       // 0x0
	.size	var_simple, 4

	.type	var_got,@object         // @var_got
	.globl	var_got
	.align	2
var_got:
	.word	0                       // 0x0
	.size	var_got, 4

	.type	var_tlsgd,@object       // @var_tlsgd
	.section	.tbss,"awT",@nobits
	.globl	var_tlsgd
	.align	2
var_tlsgd:
	.word	0                       // 0x0
	.size	var_tlsgd, 4

	.type	var_tlsld,@object       // @var_tlsld
	.globl	var_tlsld
	.align	2
var_tlsld:
	.word	0                       // 0x0
	.size	var_tlsld, 4

	.type	var_tlsie,@object       // @var_tlsie
	.globl	var_tlsie
	.align	2
var_tlsie:
	.word	0                       // 0x0
	.size	var_tlsie, 4

	.type	var_tlsle,@object       // @var_tlsle
	.globl	var_tlsle
	.align	2
var_tlsle:
	.word	0                       // 0x0
	.size	var_tlsle, 4


