# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvmac %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.mac t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.mac t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.mac 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.mac t0, t1
# CHECK-ERROR: too few operands for instruction

cv.mac t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.machhsn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.machhsn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.machhsrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhsrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhsrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.machhsrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.machhun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.machhun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.machhurn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhurn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhurn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.machhurn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhurn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhurn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.machhurn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.machhurn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.macsn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.macsn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.macsrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macsrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macsrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.macsrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.macun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.macun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.macun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.macurn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macurn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macurn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.macurn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.macurn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macurn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.macurn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.macurn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.msu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.msu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.msu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.msu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.msu t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.mulhhs t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhs t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulhhs 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulhhs t0, t1
# CHECK-ERROR: too few operands for instruction

cv.mulhhs t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulhhsn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhsrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhsrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulhhsrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulhhu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulhhu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.mulhhu t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.mulhhun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulhhun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhurn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhurn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhurn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulhhurn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhurn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhurn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulhhurn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulhhurn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.muls t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.muls t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.muls 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.muls t0, t1
# CHECK-ERROR: too few operands for instruction

cv.muls t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.mulsn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulsn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulsrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulsrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulsrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulsrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.mulu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.mulu t0, t1, t2, t4
# CHECK-ERROR: invalid operand for instruction

cv.mulun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.mulurn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulurn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulurn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.mulurn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulurn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulurn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.mulurn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.mulurn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction
