# RUN: not llvm-mc -triple riscv32 -mattr=+zilsd < %s 2>&1 | FileCheck %s

# Out of range immediates
## simm12
ld t1, -2049(a0) # CHECK: :[[@LINE]]:8: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
sd t1, 2048(a0) # CHECK: :[[@LINE]]:8: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]

# Invalid register names
ld t2, (4)a0 # CHECK: :[[@LINE]]:4: error: register must be even
ld s3, (4)a0 # CHECK: :[[@LINE]]:4: error: register must be even
sd t2, (10)s2 # CHECK: :[[@LINE]]:4: error: register must be even
sd a7, (10)s2 # CHECK: :[[@LINE]]:4: error: register must be even
ld x0, a_symbol # CHECK: :[[@LINE]]:4: error: register pair must start with an even GPR other than x0
sd a2, a_symbol, x0 # CHECK: :[[@LINE]]:18: error: register must be a GPR excluding zero (x0)
