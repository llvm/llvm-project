# RUN: llvm-mc -triple i386-unknown-unknown %s -I  %p | FileCheck %s

# CHECK: TESTA:
# CHECK: TEST0:
# CHECK: a = 0
# CHECK: TESTB:
TESTA:  
	.include       "directive_set.s"
TESTB:
