# RUN: llvm-mc --verify -triple riscv32 -mattr=+zba < %s

# Too few operands
sh1add t0, t1 # expected-error:14 {{too few operands for instruction}}
# Too few operands
sh2add t0, t1 # expected-error:14 {{too few operands for instruction}}
# Too few operands
sh3add t0, t1 # expected-error:14 {{too few operands for instruction}}
slli.uw t0, t1, 0 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
add.uw t0, t1, t2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
sh1add.uw t0, t1, t2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
sh2add.uw t0, t1, t2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
sh3add.uw t0, t1, t2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
