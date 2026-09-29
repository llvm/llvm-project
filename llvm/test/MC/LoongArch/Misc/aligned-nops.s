# RUN: llvm-mc --filetype=obj --triple=loongarch64 < %s \
# RUN:     | llvm-objdump -d - | FileCheck %s

# func1 and func2 are 8 byte alignment but the func1's size is 4.
# So assembler will insert a nop to make sure 8 byte alignment.

.text

.p2align 3
func1:
              addi.d $sp, $sp, -16
# CHECK:      addi.d $sp, $sp, -16
# CHECK-NEXT: nop
.p2align 3
func2:
