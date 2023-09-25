## Check that we can use an absolute symbol value as a CSR number
# RUN: llvm-mc -triple riscv32 %s | FileCheck %s
# RUN: not llvm-mc -triple riscv32 %s --defsym=CHECK_BAD=1 2>&1 | FileCheck %s --check-prefix=ERROR

.set fflags_abs_sym, 1
csrr a0, fflags_abs_sym
# CHECK: csrr	a0, fflags

csrr a0, (fflags_abs_sym+1)
# CHECK: csrr	a0, frm

.equ fplus_one_abs_sym, fflags_abs_sym + 1
csrr a0, fplus_one_abs_sym
# CHECK: csrr	a0, frm

## Check that redefining the value is allowed
## If we were to use Sym->getVariableValue(true) this code would assert with
## Assertion `!IsUsed && "Cannot set a variable that has already been used."' failed.
.set csr_index, 1
# CHECK: csrr	a0, fflags
csrr a0, csr_index
.set csr_index, 2
# CHECK: csrr	a0, frm
csrr a0, csr_index

.ifdef CHECK_BAD
.set out_of_range, 4097
csrr a0, out_of_range
# ERROR: [[#@LINE-1]]:10: error: operand must be a valid system register name or an integer in the range [0, 4095]

csrr a0, undef_symbol
# ERROR: [[#@LINE-1]]:10: error: operand must be a valid system register name or an integer in the range [0, 4095]

local_label:
csrr a0, local_label
# ERROR: [[#@LINE-1]]:10: error: operand must be a valid system register name or an integer in the range [0, 4095]

.Lstart:
.space 10
.Lend:
csrr a0, .Lstart-.Lend
# ERROR: [[#@LINE-1]]:10: error: operand must be a valid system register name or an integer in the range [0, 4095]

.set dot_set_sym_diff, .Lstart-.Lend
csrr a0, dot_set_sym_diff
# ERROR: [[#@LINE-1]]:10: error: operand must be a valid system register name or an integer in the range [0, 4095]

.endif
