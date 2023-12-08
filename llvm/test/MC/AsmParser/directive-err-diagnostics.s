// RUN: not llvm-mc -triple i386 %s 2>&1 | FileCheck %s

	.error 0
// CHECK: error: .error argument must be a string
// CHECK:	.error 0
// CHECK:              ^

	.ifeqs "0", "1"
		.ifeqs "", ""
			.error "message"
		.endif
	.endif
// CHECK-NOT: error: message
// CHECK-NOT: error: invalid instruction mnemonic 'message'
// CHECK-NOT:	.error "message"
// CHECK-NOT:          ^

