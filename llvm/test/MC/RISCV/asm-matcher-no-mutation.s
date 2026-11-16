# REQUIRES: asserts
# RUN: not llvm-mc -triple=riscv64 -mattr=+xsmtvdotii -debug-only=asm-matcher %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=VMADOT
# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-y -debug-only=asm-matcher %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=YADD
# RUN: not llvm-mc -triple=riscv32 -debug-only=asm-matcher %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=RV32

## Verify whether validateTargetOperandClass or custom operand parsing mutates
## parsed register operands across match table rows.

## TODO: validateTargetOperandClass should not mutate v0 into V0M2 (356) or v3
## into noreg (0) when SMT_VMADOT1 fails to match.
# VMADOT:      Trying to match opcode SMT_VMADOT1
# VMADOT-NEXT:   Matching formal operand class MCK_VRM2 against actual operand at index 1 (<reg: v0 (35)>): match success using target matcher
# VMADOT-NEXT:   Matching formal operand class MCK_VRM2 against actual operand at index 2 (<reg: v3 (38)>): operand match failed, recording near-miss with diag code 0
# VMADOT:      Trying to match opcode SMT_VMADOT1vtype
# VMADOT-NEXT:   Matching formal operand class MCK_VRM2 against actual operand at index 1 (<reg: v0 (356)>): match success using generic matcher
# VMADOT-NEXT:   Matching formal operand class MCK_VRM2 against actual operand at index 2 (<reg: noreg (0)>): operand match failed, recording near-miss with diag code 0
smt.vmadot1 v0, v3, v4

## TODO: validateTargetOperandClass should not mutate a0/a1 from GPR (77/78)
## into YGPR (302/303) when YADD fails to match.
# YADD:      Trying to match opcode YADD
# YADD-NEXT:   Matching formal operand class MCK_YGPR against actual operand at index 1 (<reg: a0 (77)>): match success using target matcher
# YADD-NEXT:   Matching formal operand class MCK_YGPR against actual operand at index 2 (<reg: a1 (78)>): match success using target matcher
# YADD-NEXT:   Matching formal operand class MCK_GPRNoX0 against actual operand at index 3 (<reg: zero (67)>): operand match failed, recording near-miss
# YADD:      Trying to match opcode YADDI
# YADD-NEXT:   Matching formal operand class MCK_YGPR against actual operand at index 1 (<reg: a0 (302)>): match success using generic matcher
# YADD-NEXT:   Matching formal operand class MCK_YGPR against actual operand at index 2 (<reg: a1 (303)>): match success using generic matcher
yadd a0, a1, x0

## TODO: parseGPRPair should not mutate a0 from GPR (77) into GPRPair (388)
## before instruction matching, which prevents matching MCK_GPR on LD.
# RV32:      Trying to match opcode LD
# RV32-NEXT:   Matching formal operand class MCK_GPR against actual operand at index 1 (<reg: a0 (388)>): operand match failed, recording near-miss
# RV32:      Trying to match opcode LD_RV32
# RV32-NEXT:   Matching formal operand class MCK_GPRPairRV32 against actual operand at index 1 (<reg: a0 (388)>): match success using generic matcher
ld a0, 0(a3)

## TODO: parseGPRPairAsFPR64 should not mutate a0/a2/a4 from GPR (77/79/81) into
## GPRPair (388/389/390) before instruction matching.
# RV32:      Trying to match opcode FADD_D
# RV32-NEXT:   Matching formal operand class MCK_FPR64 against actual operand at index 1 (<reg: a0 (388) GPRasFPR>): operand match failed, recording near-miss
# RV32:      Trying to match opcode FADD_D_INX
# RV32-NEXT:   Matching formal operand class MCK_GPRF64AsFPR against actual operand at index 1 (<reg: a0 (388) GPRasFPR>): operand match failed, recording near-miss
# RV32:      Trying to match opcode FADD_D_IN32X
# RV32-NEXT:   Matching formal operand class MCK_GPRPairAsFPR against actual operand at index 1 (<reg: a0 (388) GPRasFPR>): match success using generic matcher
# RV32-NEXT:   Matching formal operand class MCK_GPRPairAsFPR against actual operand at index 2 (<reg: a2 (389) GPRasFPR>): match success using generic matcher
# RV32-NEXT:   Matching formal operand class MCK_GPRPairAsFPR against actual operand at index 3 (<reg: a4 (390) GPRasFPR>): match success using generic matcher
fadd.d a0, a2, a4
