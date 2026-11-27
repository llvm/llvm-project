# RUN: llvm-mc -triple=xtensa -mcpu=esp32s3 -filetype=obj %s -o %t
# RUN: llvm-objdump -d --mcpu=esp32s3 --no-show-raw-insn %t | FileCheck %s

# select_2
ee.fft.ams.s16.ld.r32.decp q0, a3, q1, q2, q3, q4, q5, 0
# CHECK: ee.fft.ams.s16.ld.r32.decp q0, a3, q1, q2, q3, q4, q5, 0
ee.fft.ams.s16.ld.r32.decp q0, a3, q1, q2, q3, q4, q5, 1
# CHECK: ee.fft.ams.s16.ld.r32.decp q0, a3, q1, q2, q3, q4, q5, 1

# select_4
ee.fft.r2bf.s16.st.incp q0, q1, q2, a3, 0
# CHECK: ee.fft.r2bf.s16.st.incp q0, q1, q2, a3, 0
ee.fft.r2bf.s16.st.incp q0, q1, q2, a3, 3
# CHECK: ee.fft.r2bf.s16.st.incp q0, q1, q2, a3, 3

# select_8
ee.ldxq.32 q0, q1, a3, 0, 0
# CHECK: ee.ldxq.32 q0, q1, a3, 0, 0
ee.ldxq.32 q0, q1, a3, 3, 7
# CHECK: ee.ldxq.32 q0, q1, a3, 3, 7

# select_16
ee.slci.2q q0, q1, 0
# CHECK: ee.slci.2q q0, q1, 0
ee.slci.2q q0, q1, 15
# CHECK: ee.slci.2q q0, q1, 15
