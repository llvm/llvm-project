# RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=ERR
	
.text

test2:	
	jmp baz
# ERR: [[@LINE+1]]:5: error: expected absolute expression
.if . - text2 == 1
	nop
.else
	ret
.endif
	push fs

# No additional errors.
#	
# 	ERR-NOT: {{[0-9]+}}:{{[0-9]+}}: error:	
