# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

# expected-error@+1:7 {{out of range literal value}}
.byte 0xffa
# expected-error@+1:7 {{out of range literal value}}
.half 0xffffa
# expected-error@+1:8 {{out of range literal value}}
.short 0xffffa
# expected-error@+1:8 {{out of range literal value}}
.hword 0xffffa
# expected-error@+1:8 {{out of range literal value}}
.2byte 0xffffa
# expected-error@+1:7 {{out of range literal value}}
.word 0xffffffffa
# expected-error@+1:7 {{out of range literal value}}
.long 0xffffffffa
# expected-error@+1:8 {{out of range literal value}}
.4byte 0xffffffffa
# expected-error@+1:8 {{literal value out of range for directive}}
.dword 0xffffffffffffffffa
# expected-error@+1:8 {{literal value out of range for directive}}
.8byte 0xffffffffffffffffa
