# RUN: llvm-mc -triple=xtensa -mcpu=esp32s3 -filetype=obj %s -o %t
# RUN: llvm-objdump -d --mcpu=esp32s3 --no-show-raw-insn %t | FileCheck %s

# offset_16_16
ee.ldf.128.ip f0, f1, f2, f3, a4, -128
# CHECK: ee.ldf.128.ip f0, f1, f2, f3, a4, -128
ee.ldf.128.ip f0, f1, f2, f3, a4, -16
# CHECK: ee.ldf.128.ip f0, f1, f2, f3, a4, -16
ee.ldf.128.ip f0, f1, f2, f3, a4, 112
# CHECK: ee.ldf.128.ip f0, f1, f2, f3, a4, 112

# offset_256_8
ee.ld.accx.ip a0, -1024
# CHECK: ee.ld.accx.ip a0, -1024
ee.ld.accx.ip a0, -8
# CHECK: ee.ld.accx.ip a0, -8
ee.ld.accx.ip a0, 1016
# CHECK: ee.ld.accx.ip a0, 1016

# offset_256_16
ee.ldqa.s16.128.ip a0, -2048
# CHECK: ee.ldqa.s16.128.ip a0, -2048
ee.ldqa.s16.128.ip a0, -16
# CHECK: ee.ldqa.s16.128.ip a0, -16
ee.ldqa.s16.128.ip a0, 2032
# CHECK: ee.ldqa.s16.128.ip a0, 2032

# offset_256_4
ee.ld.qacc_h.h.32.ip a0, -512
# CHECK: ee.ld.qacc_h.h.32.ip a0, -512
ee.ld.qacc_h.h.32.ip a0, -4
# CHECK: ee.ld.qacc_h.h.32.ip a0, -4
ee.ld.qacc_h.h.32.ip a0, 508
# CHECK: ee.ld.qacc_h.h.32.ip a0, 508

# offset_128_2
ee.vldbc.16.ip q0, a0, 0
# CHECK: ee.vldbc.16.ip q0, a0, 0
ee.vldbc.16.ip q0, a0, 4
# CHECK: ee.vldbc.16.ip q0, a0, 4
ee.vldbc.16.ip q0, a0, 254
# CHECK: ee.vldbc.16.ip q0, a0, 254

# offset_64_16
ee.vmulas.s16.accx.ld.ip q0, a0, -512, q1, q2
# CHECK: ee.vmulas.s16.accx.ld.ip q0, a0, -512, q1, q2
ee.vmulas.s16.accx.ld.ip q0, a0, -16, q1, q2
# CHECK: ee.vmulas.s16.accx.ld.ip q0, a0, -16, q1, q2
ee.vmulas.s16.accx.ld.ip q0, a0, 496, q1, q2
# CHECK: ee.vmulas.s16.accx.ld.ip q0, a0, 496, q1, q2
