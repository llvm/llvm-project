// RUN: not llvm-mc %s -triple=aarch64-none-linux-gnu -filetype asm -o - 2>&1 \
// RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.align 2
	.global diagnostics
	.type diagnostics,%function
diagnostics:
.Label:
    .inst
// CHECK-ERROR: expected expression following directive

    .inst 0x5e104020,
// CHECK-ERROR: expected expression

    .inst .Label
// CHECK-ERROR: expected constant expression

    .inst 0x5e104020 0x5e104020
// CHECK-ERROR: unexpected token in directive

// CHECK-ERROR-NOT: unexpected token at start of statement	
