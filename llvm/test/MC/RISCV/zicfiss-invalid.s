# RUN: llvm-mc --verify %s -triple=riscv32 -mattr=+zicfiss,+zcmop,+c -M no-aliases -show-encoding
# RUN: llvm-mc --verify %s -triple=riscv64 -mattr=+zicfiss,+zcmop,+c -M no-aliases -show-encoding

# expected-error@+1 {{register must be ra or t0 (x1 or x5)}}
sspopchk a1

# expected-error@+1 {{register must be ra (x1)}}
c.sspush t0

# expected-error@+1 {{register must be t0 (x5)}}
c.sspopchk ra

# expected-error@+1 {{register must be ra or t0 (x1 or x5)}}
sspush a0

# expected-error@+1 {{register must be a GPR excluding zero (x0)}}
ssrdp zero
