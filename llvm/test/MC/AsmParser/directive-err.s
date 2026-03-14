// RUN: not llvm-mc -triple i386 %s 2>&1 | FileCheck %s

	.err
// CHECK: error: .err encountered
// CHECK-NEXT: 	.err
// CHECK-NEXT:  ^

	.ifc a,a
		.err
	.endif
// CHECK: error: .err encountered
// CHECK-NEXT:		.err
// CHECK-NEXT:          ^

	.ifnc a,a
		.err
	.endif
// CHECK-NOT: error: .err encountered

	.error "This is my error.  There are many like it, but this one is mine."
// CHECK: error: This is my error.  There are many like it, but this one is mine.

	.ifc one, two
		.error "My error is my best friend."
	.endif
// CHECK-NOT: error: My error is my best friend.

	.error
// CHECK: error: .error directive invoked in source file

