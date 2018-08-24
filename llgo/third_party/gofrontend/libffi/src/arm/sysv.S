/* -----------------------------------------------------------------------
   sysv.S - Copyright (c) 1998, 2008, 2011 Red Hat, Inc.
	    Copyright (c) 2011 Plausible Labs Cooperative, Inc.

   ARM Foreign Function Interface

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   ``Software''), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED ``AS IS'', WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
   ----------------------------------------------------------------------- */

#define LIBFFI_ASM	
#include <fficonfig.h>
#include <ffi.h>
#include <ffi_cfi.h>
#include "internal.h"

/* GCC 4.8 provides __ARM_ARCH; construct it otherwise.  */
#ifndef __ARM_ARCH
# if defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) \
     || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) \
     || defined(__ARM_ARCH_7EM__)
#  define __ARM_ARCH 7
# elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) \
        || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) \
        || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6T2__) \
	|| defined(__ARM_ARCH_6M__)
#  define __ARM_ARCH 6
# elif defined(__ARM_ARCH_5__) || defined(__ARM_ARCH_5T__) \
	|| defined(__ARM_ARCH_5E__) || defined(__ARM_ARCH_5TE__) \
	|| defined(__ARM_ARCH_5TEJ__)
#  define __ARM_ARCH 5
# else
#  define __ARM_ARCH 4
# endif
#endif

/* Conditionally compile unwinder directives.  */
.macro UNWIND text:vararg
#ifdef __ARM_EABI__
	\text
#endif	
.endm
#if defined(HAVE_AS_CFI_PSEUDO_OP) && defined(__ARM_EABI__)
	.cfi_sections	.debug_frame
#endif

#define CONCAT(a, b)	CONCAT2(a, b)
#define CONCAT2(a, b)	a ## b

#ifdef __USER_LABEL_PREFIX__
# define CNAME(X)	CONCAT (__USER_LABEL_PREFIX__, X)
#else
# define CNAME(X)	X
#endif
#ifdef __ELF__
# define SIZE(X)	.size CNAME(X), . - CNAME(X)
# define TYPE(X, Y)	.type CNAME(X), Y
#else
# define SIZE(X)
# define TYPE(X, Y)
#endif

#define ARM_FUNC_START(name, gl) \
	.align	3; \
	.ifne gl; .globl CNAME(name); FFI_HIDDEN(CNAME(name)); .endif; \
	TYPE(name, %function); \
	CNAME(name):

#define ARM_FUNC_END(name) \
	SIZE(name)

/* Aid in defining a jump table with 8 bytes between entries.  */
.macro E index
	.if . - 0b - 8*\index
	.error "type table out of sync"
	.endif
.endm

	.text
	.syntax unified
	.arm

	/* We require interworking on LDM, which implies ARMv5T,
	   which implies the existance of BLX.  */
	.arch	armv5t

	/* Note that we use STC and LDC to encode VFP instructions,
	   so that we do not need ".fpu vfp", nor get that added to
	   the object file attributes.  These will not be executed
	   unless the FFI_VFP abi is used.  */

	@ r0:   stack
	@ r1:   frame
	@ r2:   fn
	@ r3:	vfp_used

ARM_FUNC_START(ffi_call_VFP, 1)
	UNWIND	.fnstart
	cfi_startproc

	cmp	r3, #3			@ load only d0 if possible
	ldcle	p11, cr0, [r0]		@ vldrle d0, [sp]
	ldcgt	p11, cr0, [r0], {16}	@ vldmgt sp, {d0-d7}
	add	r0, r0, #64		@ discard the vfp register args
	/* FALLTHRU */
ARM_FUNC_END(ffi_call_VFP)

ARM_FUNC_START(ffi_call_SYSV, 1)
	stm	r1, {fp, lr}
	mov	fp, r1

	@ This is a bit of a lie wrt the origin of the unwind info, but
	@ now we've got the usual frame pointer and two saved registers.
	UNWIND	.save {fp,lr}
	UNWIND	.setfp fp, sp
	cfi_def_cfa(fp, 8)
	cfi_rel_offset(fp, 0)
	cfi_rel_offset(lr, 4)

	mov	sp, r0		@ install the stack pointer
	mov	lr, r2		@ move the fn pointer out of the way
	ldr	ip, [fp, #16]	@ install the static chain
	ldmia	sp!, {r0-r3}	@ move first 4 parameters in registers.
	blx	lr		@ call fn

	@ Load r2 with the pointer to storage for the return value
	@ Load r3 with the return type code
	ldr	r2, [fp, #8]
	ldr	r3, [fp, #12]

	@ Deallocate the stack with the arguments.
	mov	sp, fp
	cfi_def_cfa_register(sp)

	@ Store values stored in registers.
	.align	3
	add	pc, pc, r3, lsl #3
	nop
0:
E ARM_TYPE_VFP_S
	stc	p10, cr0, [r2]		@ vstr s0, [r2]
	pop	{fp,pc}
E ARM_TYPE_VFP_D
	stc	p11, cr0, [r2]		@ vstr d0, [r2]
	pop	{fp,pc}
E ARM_TYPE_VFP_N
	stc	p11, cr0, [r2], {8}	@ vstm r2, {d0-d3}
	pop	{fp,pc}
E ARM_TYPE_INT64
	str	r1, [r2, #4]
	nop
E ARM_TYPE_INT
	str	r0, [r2]
	pop	{fp,pc}
E ARM_TYPE_VOID
	pop	{fp,pc}
	nop
E ARM_TYPE_STRUCT
	pop	{fp,pc}

	cfi_endproc
	UNWIND	.fnend
ARM_FUNC_END(ffi_call_SYSV)


/*
	int ffi_closure_inner_* (cif, fun, user_data, frame)
*/

ARM_FUNC_START(ffi_go_closure_SYSV, 1)
	cfi_startproc
	stmdb	sp!, {r0-r3}			@ save argument regs
	cfi_adjust_cfa_offset(16)
	ldr	r0, [ip, #4]			@ load cif
	ldr	r1, [ip, #8]			@ load fun
	mov	r2, ip				@ load user_data
	b	0f
	cfi_endproc
ARM_FUNC_END(ffi_go_closure_SYSV)

ARM_FUNC_START(ffi_closure_SYSV, 1)
	UNWIND	.fnstart
	cfi_startproc
	stmdb	sp!, {r0-r3}			@ save argument regs
	cfi_adjust_cfa_offset(16)
	ldr	r0, [ip, #FFI_TRAMPOLINE_SIZE]	  @ load cif
	ldr	r1, [ip, #FFI_TRAMPOLINE_SIZE+4]  @ load fun
	ldr	r2, [ip, #FFI_TRAMPOLINE_SIZE+8]  @ load user_data
0:
	add	ip, sp, #16			@ compute entry sp
	sub	sp, sp, #64+32			@ allocate frame
	cfi_adjust_cfa_offset(64+32)
	stmdb	sp!, {ip,lr}

	/* Remember that EABI unwind info only applies at call sites.
	   We need do nothing except note the save of the stack pointer
	   and the link registers.  */
	UNWIND	.save {sp,lr}
	cfi_adjust_cfa_offset(8)
	cfi_rel_offset(lr, 4)

	add	r3, sp, #8			@ load frame
	bl	CNAME(ffi_closure_inner_SYSV)

	@ Load values returned in registers.
	add	r2, sp, #8+64			@ load result
	adr	r3, CNAME(ffi_closure_ret)
	add	pc, r3, r0, lsl #3
	cfi_endproc
	UNWIND	.fnend
ARM_FUNC_END(ffi_closure_SYSV)

ARM_FUNC_START(ffi_go_closure_VFP, 1)
	cfi_startproc
	stmdb	sp!, {r0-r3}			@ save argument regs
	cfi_adjust_cfa_offset(16)
	ldr	r0, [ip, #4]			@ load cif
	ldr	r1, [ip, #8]			@ load fun
	mov	r2, ip				@ load user_data
	b	0f
	cfi_endproc
ARM_FUNC_END(ffi_go_closure_VFP)

ARM_FUNC_START(ffi_closure_VFP, 1)
	UNWIND	.fnstart
	cfi_startproc
	stmdb	sp!, {r0-r3}			@ save argument regs
	cfi_adjust_cfa_offset(16)
	ldr	r0, [ip, #FFI_TRAMPOLINE_SIZE]	  @ load cif
	ldr	r1, [ip, #FFI_TRAMPOLINE_SIZE+4]  @ load fun
	ldr	r2, [ip, #FFI_TRAMPOLINE_SIZE+8]  @ load user_data
0:
	add	ip, sp, #16
	sub	sp, sp, #64+32			@ allocate frame
	cfi_adjust_cfa_offset(64+32)
	stc	p11, cr0, [sp], {16}		@ vstm sp, {d0-d7}
	stmdb	sp!, {ip,lr}

	/* See above.  */
	UNWIND	.save {sp,lr}
	cfi_adjust_cfa_offset(8)
	cfi_rel_offset(lr, 4)

	add	r3, sp, #8			@ load frame
	bl	CNAME(ffi_closure_inner_VFP)

	@ Load values returned in registers.
	add	r2, sp, #8+64			@ load result
	adr	r3, CNAME(ffi_closure_ret)
	add	pc, r3, r0, lsl #3
	cfi_endproc
	UNWIND	.fnend
ARM_FUNC_END(ffi_closure_VFP)

/* Load values returned in registers for both closure entry points.
   Note that we use LDM with SP in the register set.  This is deprecated
   by ARM, but not yet unpredictable.  */

ARM_FUNC_START(ffi_closure_ret, 0)
	cfi_startproc
	cfi_rel_offset(sp, 0)
	cfi_rel_offset(lr, 4)
0:
E ARM_TYPE_VFP_S
	ldc	p10, cr0, [r2]			@ vldr s0, [r2]
	ldm	sp, {sp,pc}
E ARM_TYPE_VFP_D
	ldc	p11, cr0, [r2]			@ vldr d0, [r2]
	ldm	sp, {sp,pc}
E ARM_TYPE_VFP_N
	ldc	p11, cr0, [r2], {8}		@ vldm r2, {d0-d3}
	ldm	sp, {sp,pc}
E ARM_TYPE_INT64
	ldr	r1, [r2, #4]
	nop
E ARM_TYPE_INT
	ldr	r0, [r2]
	ldm	sp, {sp,pc}
E ARM_TYPE_VOID
	ldm	sp, {sp,pc}
	nop
E ARM_TYPE_STRUCT
	ldm	sp, {sp,pc}
	cfi_endproc
ARM_FUNC_END(ffi_closure_ret)

#if FFI_EXEC_TRAMPOLINE_TABLE

/* ??? The iOS support should be updated.  The first insn used to
   be STMFD, but that's been moved into ffi_closure_SYSV.  If the
   writable page is put after this one we can make use of the
   pc+8 feature of the architecture.  We can also reduce the size
   of the thunk to 8 and pack more of these into the page.

   In the meantime, simply replace the STMFD with a NOP so as to
   keep all the magic numbers the same within ffi.c.  */

	.align	12
ARM_FUNC_START(ffi_closure_trampoline_table_page)
.rept	4096 / 12
	nop
	ldr	ip, [pc, #-4092]
	ldr	pc, [pc, #-4092]
.endr

#else

ARM_FUNC_START(ffi_arm_trampoline, 1)
0:	adr	ip, 0b
	ldr	pc, 1f
1:	.long	0
ARM_FUNC_END(ffi_arm_trampoline)

#endif /* FFI_EXEC_TRAMPOLINE_TABLE */

#if defined __ELF__ && defined __linux__
	.section	.note.GNU-stack,"",%progbits
#endif
