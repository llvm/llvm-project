# RUN: llvm-mc --verify %s -triple=riscv32 -mattr=+zicfiss,+zcmop,+c -M no-aliases -show-encoding
# RUN: llvm-mc --verify %s -triple=riscv64 -mattr=+zicfiss,+zcmop,+c -M no-aliases -show-encoding

sspopchk a1 # expected-error {{register must be ra or t0 (x1 or x5)}}

c.sspush t0 # expected-error {{register must be ra (x1)}}

c.sspopchk ra # expected-error {{register must be t0 (x5)}}

sspush a0 # expected-error {{register must be ra or t0 (x1 or x5)}}

ssrdp zero # expected-error {{register must be a GPR excluding zero (x0)}}
