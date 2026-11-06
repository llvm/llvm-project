# RUN: llvm-mc --verify -triple riscv32 < %s

call 1234 # expected-error:6 {{operand must be a bare symbol name}}
call %pcrel_hi(1234) # expected-error:6 {{operand must be a bare symbol name}}
call %pcrel_lo(1234) # expected-error:6 {{operand must be a bare symbol name}}
call %pcrel_hi(foo) # expected-error:6 {{operand must be a bare symbol name}}
call %pcrel_lo(foo) # expected-error:6 {{operand must be a bare symbol name}}
call %hi(1234) # expected-error:6 {{operand must be a bare symbol name}}
call %lo(1234) # expected-error:6 {{operand must be a bare symbol name}}
call %hi(foo) # expected-error:6 {{operand must be a bare symbol name}}
call %lo(foo) # expected-error:6 {{operand must be a bare symbol name}}
call foo, bar # expected-error:6 {{register must be a GPR}}
call t0, %lo(foo) # expected-error:10 {{operand must be a bare symbol name}}
call foo@pls # expected-error:10 {{@ (except the deprecated/ignored @plt) is disallowed}}
call foo@3 # expected-error:10 {{@ (except the deprecated/ignored @plt) is disallowed}}
