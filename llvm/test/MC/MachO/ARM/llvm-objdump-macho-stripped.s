@ RUN: llvm-mc -triple=thumbv7-apple-ios -filetype=obj -o - < %s | llvm-objdump -d --macho --triple=thumbv7-apple-ios - | FileCheck %s
	nop
# CHECK:        0:	00 bf                                        	nop
# We are checking that disassembly happens when there are no symbols.
# rdar://11460289
