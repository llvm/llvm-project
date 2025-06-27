# RUN: llvm-mc -triple i386-unknown-unknown %s -I  %p | FileCheck %s

# CHECK: TESTA:
# CHECK: TEST0:
# CHECK: a = 0
# CHECK: TESTB:
TESTA:  
	.include       "directive\137set.s"   # "\137" is underscore "_"
TESTB:
