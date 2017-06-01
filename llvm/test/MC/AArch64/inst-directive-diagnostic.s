// RUN: not llvm-mc %s -triple=aarch64-none-linux-gnu -filetype asm -o - 2>&1 \
// RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.align 2
	.global diagnostics
	.type diagnostics,%function
diagnostics:
.Label:
    .inst
// CHECK: [[@LINE-1]]:5: error: expected expression following '.inst' directive 

    .inst 0x5e104020,
// CHECK: [[@LINE-1]]:22: error: unknown token in expression in '.inst' directive

    .inst .Label
// CHECK: [[@LINE-1]]:11: error: expected constant expression in '.inst' directive

    .inst 0x5e104020 0x5e104020
// CHECK: [[@LINE-1]]:22: error: unexpected token in '.inst' directive

// CHECK-ERROR-NOT: unexpected token at start of statement	
