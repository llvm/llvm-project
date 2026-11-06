# RUN: llvm-mc --verify %s -triple=riscv64 -mattr=+zmmul -M no-aliases

# expected-error@+1 {{instruction requires the following: 'M' (Integer Multiplication and Division)}}
divw tp, t0, t1

# expected-error@+1 {{instruction requires the following: 'M' (Integer Multiplication and Division)}}
divuw t2, s0, s2

# expected-error@+1 {{instruction requires the following: 'M' (Integer Multiplication and Division)}}
remw a0, a1, a2

# expected-error@+1 {{instruction requires the following: 'M' (Integer Multiplication and Division)}}
remuw a3, a4, a5
