# RUN: llvm-mc --verify -filetype=obj %s -triple=riscv32 -o /dev/null
# RUN: llvm-mc --verify -filetype=obj %s -triple=riscv64 -o /dev/null

.byte foo   # expected-error:7 {{1-byte data relocations not supported}}
.2byte foo  # expected-error:8 {{2-byte data relocations not supported}}

# Test that using li with a symbol difference constant rejects values that
# cannot fit in a signed 12-bit integer.
.Lbuf: .skip (1 << 11)
.Lbuf_end:
.equ CONST, .Lbuf_end - .Lbuf
li a0, CONST # expected-error {{operand must be a constant 12-bit integer}}
li a0, .Lbuf_end - .Lbuf # expected-error {{operand must be a constant 12-bit integer}}
