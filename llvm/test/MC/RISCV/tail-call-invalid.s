# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

tail 1234 # expected-error:6 {{operand must be a bare symbol name}}
tail %pcrel_hi(1234) # expected-error:6 {{operand must be a bare symbol name}}
tail %pcrel_lo(1234) # expected-error:6 {{operand must be a bare symbol name}}
tail %pcrel_hi(foo) # expected-error:6 {{operand must be a bare symbol name}}
tail %pcrel_lo(foo) # expected-error:6 {{operand must be a bare symbol name}}
tail %hi(1234) # expected-error:6 {{operand must be a bare symbol name}}
tail %lo(1234) # expected-error:6 {{operand must be a bare symbol name}}
tail %hi(foo) # expected-error:6 {{operand must be a bare symbol name}}
tail %lo(foo) # expected-error:6 {{operand must be a bare symbol name}}
tail foo@pls # expected-error:10 {{@ (except the deprecated/ignored @plt) is disallowed}}
