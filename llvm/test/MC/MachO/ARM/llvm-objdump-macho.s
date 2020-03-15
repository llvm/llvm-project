@ RUN: llvm-mc -triple=thumbv7-apple-ios -filetype=obj -o - < %s | llvm-objdump -d --macho --triple=thumbv7-apple-ios - | FileCheck %s
.thumb
.thumb_func _fib
_fib:
	push	{r7, lr}
	pop	{r7, pc}
.thumb_func _main
_main:
	push	{r7, lr}
        pop	{r7, pc}
	nop
# CHECK: _fib:
# CHECK:        0:	80 b5                                        	push	{r7, lr}
# CHECK:        2:	80 bd                                        	pop	{r7, pc}
# CHECK: _main:
# CHECK:        4:	80 b5                                        	push	{r7, lr}
# CHECK:        6:	80 bd                                        	pop	{r7, pc}
# CHECK:        8:	00 bf                                        	nop
# We are checking that second function is fully disassembled.
# rdar://11426465
