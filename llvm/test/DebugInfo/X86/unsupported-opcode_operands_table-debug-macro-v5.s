## This test checks llvm-dwarfdump emits correct error diagnostics for the
## unsupported case where the opcode_operands_table flag is present in the
## debug_macro section header.

# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o -| \
# RUN:   not llvm-dwarfdump -debug-macro - /dev/null 2>&1 | FileCheck %s

# CHECK:error: opcode_operands_table is not supported

	.section	.debug_macro,"",@progbits
.Lcu_macro_begin0:
	.short	5                      # Macro information version
	.byte	4                       # Flags: 32 bit, debug_line_offset absent, opcode_operands_table present
