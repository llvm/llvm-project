// RUN: llvm-mc -triple x86_64-unknown-unknown -defsym=ERR=0 %s -o -      | FileCheck %s
// RUN: not llvm-mc -triple x86_64-unknown-unknown -defsym=ERR=1 %s -o - 2>&1 | FileCheck --check-prefix=ERR %s
	
// CHECK-NOT: .set var_xdata
var_xdata = %rcx

// CHECK: xorq %rcx, %rcx
xorq var_xdata, var_xdata

.if (ERR==1)
// ERR: [[@LINE+2]]:15: error: unknown token in expression in '.set' directive
// ERR: [[@LINE+1]]:15: error: missing expression in '.set' directive
.set err_var, %rcx
.endif	

	
