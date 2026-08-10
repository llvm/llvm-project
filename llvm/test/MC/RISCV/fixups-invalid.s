# RUN: not llvm-mc -filetype=obj %s -triple=riscv32 -o /dev/null 2>&1 \
# RUN:     | FileCheck %s
# RUN: not llvm-mc -filetype=obj %s -triple=riscv64 -o /dev/null 2>&1 \
# RUN:     | FileCheck %s

.byte foo   # CHECK: [[@LINE]]:7: error: 1-byte data relocations not supported
.2byte foo  # CHECK: [[@LINE]]:8: error: 2-byte data relocations not supported

# Test that using li with a symbol difference constant rejects values that
# cannot fit in a signed 12-bit integer.
.Lbuf: .skip (1 << 11)
.Lbuf_end:
.equ CONST, .Lbuf_end - .Lbuf
# CHECK: error: operand must be a constant 12-bit integer
li a0, CONST
# CHECK: error: operand must be a constant 12-bit integer
li a0, .Lbuf_end - .Lbuf
