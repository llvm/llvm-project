# RUN: llvm-mc -triple=x86_64-apple-darwin %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# ASM:      Lexception0:
# ASM-NEXT: 	.uleb128 Lttbase0-Lttbaseref0

# CHECK:      Contents of section __TEXT,__text:
# CHECK-NEXT:  0000 e8000000 0090e900 000000             ...........
# CHECK:      Contents of section __TEXT,__gcc_except_tab:
# CHECK-NEXT:  000b 020106                               ...

	.section	__TEXT,__text,regular,pure_instructions
Lfunc_begin0:
	callq   ___cxa_begin_catch
Ltmp1:
	nop
	jmp	___cxa_end_catch                ## TAILCALL
Lfunc_end0:
	.section	__TEXT,__gcc_except_tab
Lexception0:
	.uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
	.uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
	.uleb128 Lfunc_end0-Ltmp1               ##   Call between Ltmp1 and Lfunc_end0

Lcst_end0:
Lttbase0:

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z1hv
__Z1hv:

.subsections_via_symbols
