# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvalu %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.addrnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.addrnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.addrnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addrnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.addrnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.addun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.addun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.addun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.extbz t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.extbz 0, t1
# CHECK-ERROR: invalid operand for instruction

cv.extbz t0
# CHECK-ERROR: too few operands for instruction

cv.extbz t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.addnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.addnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.addnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.clipu t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clipu t0, t1, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clipu t0, t1, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clipu t0, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.clipu 0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.clipu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.clipu t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.minu t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.abs t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.abs 0, t1
# CHECK-ERROR: invalid operand for instruction

cv.abs t0
# CHECK-ERROR: too few operands for instruction

cv.abs t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.addrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.addrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.suburn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.suburn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.suburn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.suburn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.suburn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.suburn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.suburn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.suburn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.clip t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clip t0, t1, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clip t0, t1, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.clip t0, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.clip 0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.clip t0, t1
# CHECK-ERROR: too few operands for instruction

cv.clip t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.addunr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.addunr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.addunr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addunr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.addunr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.addurn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addurn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addurn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addurn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.addurn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addurn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addurn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.addurn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.subun t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subun t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subun t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subun t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.subun t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subun 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subun t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.subun t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.subn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.subn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.subn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.subrnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.subrnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.slet t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.slet t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.slet 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.slet t0, t1
# CHECK-ERROR: too few operands for instruction

cv.slet t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.suburnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.suburnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.suburnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.suburnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.suburnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.maxu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.maxu t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.extbs t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.extbs 0, t1
# CHECK-ERROR: invalid operand for instruction

cv.extbs t0
# CHECK-ERROR: too few operands for instruction

cv.extbs t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.exths t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.exths 0, t1
# CHECK-ERROR: invalid operand for instruction

cv.exths t0
# CHECK-ERROR: too few operands for instruction

cv.exths t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max t0, t1
# CHECK-ERROR: too few operands for instruction

cv.max t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.subunr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subunr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subunr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subunr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.subunr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.exthz t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.exthz 0, t1
# CHECK-ERROR: invalid operand for instruction

cv.exthz t0
# CHECK-ERROR: too few operands for instruction

cv.exthz t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.clipur t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.clipur t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.clipur 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.clipur t0, t1
# CHECK-ERROR: too few operands for instruction

cv.clipur t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.addurnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.addurnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.addurnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.addurnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.addurnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.addn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.addn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.addn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.addn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.addn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.subrn t0, t1, t2, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subrn t0, t1, t2, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subrn t0, t1, t2, a0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.subrn t0, t1, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrn t0, 0, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrn 0, t1, t2, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrn t0, t1, t2
# CHECK-ERROR: too few operands for instruction

cv.subrn t0, t1, t2, 0, a0
# CHECK-ERROR: invalid operand for instruction

cv.subnr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subnr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subnr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subnr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.subnr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.clipr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.clipr t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.clipr 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.clipr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.clipr t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.sletu t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sletu t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sletu 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sletu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.sletu t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction

cv.min t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min t0, t1
# CHECK-ERROR: too few operands for instruction

cv.min t0, t1, t2, a0
# CHECK-ERROR: invalid operand for instruction
