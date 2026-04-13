;; RUN:  llvm-mc -triple riscv32-apple-macho %s -filetype=obj -o - | \
;; RUN:  llvm-objdump - -d  | FileCheck %s
	
Ltmp0:	
        addi a0, a0, 1

	;; CHECK: 00000000 <ltmp0>:
