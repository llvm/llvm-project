// RUN: llvm-mc -triple i386 %s 2>&1 | FileCheck %s

	.warning
// CHECK: warning: .warning directive invoked in source file
// CHECK-NEXT: 	.warning
// CHECK-NEXT:  ^

	.ifc a,a
		.warning
	.endif
// CHECK: warning: .warning directive invoked in source file
// CHECK-NEXT:		.warning
// CHECK-NEXT:          ^

	.ifnc a,a
		.warning
	.endif
// CHECK-NOT: warning: .warning directive invoked in source file

	.warning "here be dragons"
// CHECK: warning: here be dragons

	.ifc one, two
		.warning "dragons, i say"
	.endif
// CHECK-NOT: warning: dragons, i say
