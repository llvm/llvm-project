# RUN: llvm-mc -triple x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-dwarfdump --eh-frame %t | FileCheck %s
# RUN: not llvm-mc -triple x86_64 --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ASM:      .cfi_remember_state{{$}}
# ASM-NEXT: pushq %rbx
# ASM:      .cfi_restore_state{{$}}
# ASM-NEXT: pushq %rbp

## .cfi_restore_state restores the CFA offset that .cfi_adjust_cfa_offset and
## .cfi_rel_offset are relative to.
# CHECK:      0x0: CFA=RSP+8: RIP=[CFA-8]
# CHECK-NEXT: 0x1: CFA=RSP+16: RIP=[CFA-8]
# CHECK-NEXT: 0x2: CFA=RSP+24: RBX=[CFA-24], RIP=[CFA-8]
# CHECK-NEXT: 0x3: CFA=RSP+16: RIP=[CFA-8]
# CHECK-NEXT: 0x4: CFA=RSP+8: RIP=[CFA-8]
# CHECK-NEXT: 0x5: CFA=RSP+24: RBX=[CFA-24], RIP=[CFA-8]
# CHECK-NEXT: 0x6: CFA=RSP+32: RBX=[CFA-24], RBP=[CFA-32], RIP=[CFA-8]
# CHECK-NEXT: 0x8: CFA=RSP+16: RIP=[CFA-8]
# CHECK-NEXT: 0x9: CFA=RSP+8: RIP=[CFA-8]
# CHECK-EMPTY:

f:
	.cfi_startproc
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	.cfi_remember_state
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	.cfi_rel_offset %rbx, 0
	.cfi_remember_state
	popq	%rbx
	.cfi_adjust_cfa_offset -8
	.cfi_restore %rbx
	popq	%rax
	.cfi_adjust_cfa_offset -8
	ret
	.cfi_restore_state
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	.cfi_rel_offset %rbp, 0
	ud2
	.cfi_restore_state
	popq	%rax
	.cfi_adjust_cfa_offset -8
	ret
	.cfi_endproc

.ifdef ERR
	.cfi_startproc
# ERR: [[#@LINE+1]]:2: error: no matching .cfi_remember_state
	.cfi_restore_state
	.cfi_remember_state
	.cfi_restore_state
# ERR: [[#@LINE+1]]:2: error: no matching .cfi_remember_state
	.cfi_restore_state
	.cfi_remember_state
	.cfi_endproc
.endif
