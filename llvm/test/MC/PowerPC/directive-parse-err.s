// RUN: not llvm-mc -triple powerpc-unknown-unknown %s     2>&1 | FileCheck %s
// RUN: not llvm-mc -triple powerpc-unknown-unknown %s     2>&1 | grep "error:" | count 8
// RUN: not llvm-mc -triple powerpc64-unknown-unknown %s   2>&1 | FileCheck %s
// RUN: not llvm-mc -triple powerpc64-unknown-unknown %s   2>&1 | grep "error:" | count 8
// RUN: not llvm-mc -triple powerpc64le-unknown-unknown %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple powerpc64le-unknown-unknown %s 2>&1 | grep "error:" | count 8

	// CHECK: [[@LINE+1]]:8: error: unknown token in expression in '.word' directive
	.word %
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.word # EOL COMMENT
	// CHECK: [[@LINE+1]]:10: error: unexpected token in '.word' directive
	.word 0 $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.word 0 # EOL COMMENT
	// CHECK: [[@LINE+1]]:11: error: unexpected token in '.llong' directive
	.llong 0 $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.llong 0 # EOL COMMENT
	// CHECK: [[@LINE+1]]:28: error: unexpected token in '.tc' directive
	.tc number64[TC],number64 $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.tc number64[TC],number64 # EOL COMMENT
	// CHECK: [[@LINE+1]]:15: error: unexpected token in '.machine' directive
	.machine any $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.machine any # EOL COMMENT
	// CHECK: [[@LINE+1]]:17: error: unexpected token in '.machine' directive
	.machine "any" $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.machine "any" # EOL COMMENT
	// CHECK: [[@LINE+1]]:16: error: unexpected token in '.abiversion' directive
	.abiversion 2 $ 
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.abiversion 2 # EOL COMMENT
	.type callee1, @function
callee1:
	nop
	nop
	// CHECK: [[@LINE+1]]:33: error: unexpected token in '.localentry' directive
	.localentry callee1, .-callee1 $
	// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.localentry callee1, .-callee1 # EOL COMMENT	
	// CHECK-NOT: error:	 
