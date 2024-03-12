# Adapted from https://github.com/riscvarchive/riscv-binutils-gdb/blob/1aeeeab05f3c39e2bfc6e99384490d4c7f484ba0/gas/testsuite/gas/riscv/vector-insns.s
# Golden value for this test: https://github.com/riscvarchive/riscv-binutils-gdb/blob/1aeeeab05f3c39e2bfc6e99384490d4c7f484ba0/gas/testsuite/gas/riscv/vector-insns.d
# Generated using the script: https://gist.github.com/imkiva/05facf1a51ff8abfeeeea8fa7bc307ad#file-rvvtestgen-java

# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+f,+a,+xtheadvector,+xtheadzvamo %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

th.vsetvl a0, a1, a2
# CHECK-INST: th.vsetvl	a0, a1, a2
# CHECK-ENCODING: [0x57,0xf5,0xc5,0x80]

th.vsetvli a0, a1, 0
# CHECK-INST: th.vsetvli	a0, a1, e8, m1, d1
# CHECK-ENCODING: [0x57,0xf5,0x05,0x00]

th.vsetvli a0, a1, 0x7ff
# CHECK-INST: th.vsetvli	a0, a1, 2047
# CHECK-ENCODING: [0x57,0xf5,0xf5,0x7f]

th.vsetvli a0, a1, e16,m2,d4
# CHECK-INST: th.vsetvli	a0, a1, e16, m2, d4
# CHECK-ENCODING: [0x57,0xf5,0x55,0x04]

th.vlb.v v4, (a0)
# CHECK-INST: th.vlb.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x12]

th.vlb.v v4, 0(a0)
# CHECK-INST: th.vlb.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x12]

th.vlb.v v4, (a0), v0.t
# CHECK-INST: th.vlb.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x10]

th.vlh.v v4, (a0)
# CHECK-INST: th.vlh.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x12]

th.vlh.v v4, 0(a0)
# CHECK-INST: th.vlh.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x12]

th.vlh.v v4, (a0), v0.t
# CHECK-INST: th.vlh.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x10]

th.vlw.v v4, (a0)
# CHECK-INST: th.vlw.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x12]

th.vlw.v v4, 0(a0)
# CHECK-INST: th.vlw.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x12]

th.vlw.v v4, (a0), v0.t
# CHECK-INST: th.vlw.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x10]

th.vlbu.v v4, (a0)
# CHECK-INST: th.vlbu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x02]

th.vlbu.v v4, 0(a0)
# CHECK-INST: th.vlbu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x02]

th.vlbu.v v4, (a0), v0.t
# CHECK-INST: th.vlbu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x00]

th.vlhu.v v4, (a0)
# CHECK-INST: th.vlhu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x02]

th.vlhu.v v4, 0(a0)
# CHECK-INST: th.vlhu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x02]

th.vlhu.v v4, (a0), v0.t
# CHECK-INST: th.vlhu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x00]

th.vlwu.v v4, (a0)
# CHECK-INST: th.vlwu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x02]

th.vlwu.v v4, 0(a0)
# CHECK-INST: th.vlwu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x02]

th.vlwu.v v4, (a0), v0.t
# CHECK-INST: th.vlwu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x00]

th.vle.v v4, (a0)
# CHECK-INST: th.vle.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x02]

th.vle.v v4, 0(a0)
# CHECK-INST: th.vle.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x02]

th.vle.v v4, (a0), v0.t
# CHECK-INST: th.vle.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x00]

th.vsb.v v4, (a0)
# CHECK-INST: th.vsb.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x02]

th.vsb.v v4, 0(a0)
# CHECK-INST: th.vsb.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x02]

th.vsb.v v4, (a0), v0.t
# CHECK-INST: th.vsb.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0x00]

th.vsh.v v4, (a0)
# CHECK-INST: th.vsh.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x02]

th.vsh.v v4, 0(a0)
# CHECK-INST: th.vsh.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x02]

th.vsh.v v4, (a0), v0.t
# CHECK-INST: th.vsh.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0x00]

th.vsw.v v4, (a0)
# CHECK-INST: th.vsw.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x02]

th.vsw.v v4, 0(a0)
# CHECK-INST: th.vsw.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x02]

th.vsw.v v4, (a0), v0.t
# CHECK-INST: th.vsw.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0x00]

th.vse.v v4, (a0)
# CHECK-INST: th.vse.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x02]

th.vse.v v4, 0(a0)
# CHECK-INST: th.vse.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x02]

th.vse.v v4, (a0), v0.t
# CHECK-INST: th.vse.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0x00]

th.vlsb.v v4, (a0), a1
# CHECK-INST: th.vlsb.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x1a]

th.vlsb.v v4, 0(a0), a1
# CHECK-INST: th.vlsb.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x1a]

th.vlsb.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsb.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x18]

th.vlsh.v v4, (a0), a1
# CHECK-INST: th.vlsh.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x1a]

th.vlsh.v v4, 0(a0), a1
# CHECK-INST: th.vlsh.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x1a]

th.vlsh.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsh.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x18]

th.vlsw.v v4, (a0), a1
# CHECK-INST: th.vlsw.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x1a]

th.vlsw.v v4, 0(a0), a1
# CHECK-INST: th.vlsw.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x1a]

th.vlsw.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsw.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x18]

th.vlsbu.v v4, (a0), a1
# CHECK-INST: th.vlsbu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x0a]

th.vlsbu.v v4, 0(a0), a1
# CHECK-INST: th.vlsbu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x0a]

th.vlsbu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsbu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x08]

th.vlshu.v v4, (a0), a1
# CHECK-INST: th.vlshu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x0a]

th.vlshu.v v4, 0(a0), a1
# CHECK-INST: th.vlshu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x0a]

th.vlshu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlshu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x08]

th.vlswu.v v4, (a0), a1
# CHECK-INST: th.vlswu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x0a]

th.vlswu.v v4, 0(a0), a1
# CHECK-INST: th.vlswu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x0a]

th.vlswu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlswu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x08]

th.vlse.v v4, (a0), a1
# CHECK-INST: th.vlse.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x0a]

th.vlse.v v4, 0(a0), a1
# CHECK-INST: th.vlse.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x0a]

th.vlse.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlse.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0x08]

th.vssb.v v4, (a0), a1
# CHECK-INST: th.vssb.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x0a]

th.vssb.v v4, 0(a0), a1
# CHECK-INST: th.vssb.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x0a]

th.vssb.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssb.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0x08]

th.vssh.v v4, (a0), a1
# CHECK-INST: th.vssh.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x0a]

th.vssh.v v4, 0(a0), a1
# CHECK-INST: th.vssh.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x0a]

th.vssh.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssh.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0x08]

th.vssw.v v4, (a0), a1
# CHECK-INST: th.vssw.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x0a]

th.vssw.v v4, 0(a0), a1
# CHECK-INST: th.vssw.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x0a]

th.vssw.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssw.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0x08]

th.vsse.v v4, (a0), a1
# CHECK-INST: th.vsse.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x0a]

th.vsse.v v4, 0(a0), a1
# CHECK-INST: th.vsse.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x0a]

th.vsse.v v4, (a0), a1, v0.t
# CHECK-INST: th.vsse.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0x08]

th.vlxb.v v4, (a0), v12
# CHECK-INST: th.vlxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x1e]

th.vlxb.v v4, 0(a0), v12
# CHECK-INST: th.vlxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x1e]

th.vlxb.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxb.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x1c]

th.vlxh.v v4, (a0), v12
# CHECK-INST: th.vlxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x1e]

th.vlxh.v v4, 0(a0), v12
# CHECK-INST: th.vlxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x1e]

th.vlxh.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxh.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x1c]

th.vlxw.v v4, (a0), v12
# CHECK-INST: th.vlxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x1e]

th.vlxw.v v4, 0(a0), v12
# CHECK-INST: th.vlxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x1e]

th.vlxw.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxw.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x1c]

th.vlxbu.v v4, (a0), v12
# CHECK-INST: th.vlxbu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x0e]

th.vlxbu.v v4, 0(a0), v12
# CHECK-INST: th.vlxbu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x0e]

th.vlxbu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxbu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x0c]

th.vlxhu.v v4, (a0), v12
# CHECK-INST: th.vlxhu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x0e]

th.vlxhu.v v4, 0(a0), v12
# CHECK-INST: th.vlxhu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x0e]

th.vlxhu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxhu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x0c]

th.vlxwu.v v4, (a0), v12
# CHECK-INST: th.vlxwu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x0e]

th.vlxwu.v v4, 0(a0), v12
# CHECK-INST: th.vlxwu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x0e]

th.vlxwu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxwu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x0c]

th.vlxe.v v4, (a0), v12
# CHECK-INST: th.vlxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x0e]

th.vlxe.v v4, 0(a0), v12
# CHECK-INST: th.vlxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x0e]

th.vlxe.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxe.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0x0c]

th.vsxb.v v4, (a0), v12
# CHECK-INST: th.vsxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x0e]

th.vsxb.v v4, 0(a0), v12
# CHECK-INST: th.vsxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x0e]

th.vsxb.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxb.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x0c]

th.vsxh.v v4, (a0), v12
# CHECK-INST: th.vsxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x0e]

th.vsxh.v v4, 0(a0), v12
# CHECK-INST: th.vsxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x0e]

th.vsxh.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxh.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x0c]

th.vsxw.v v4, (a0), v12
# CHECK-INST: th.vsxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x0e]

th.vsxw.v v4, 0(a0), v12
# CHECK-INST: th.vsxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x0e]

th.vsxw.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxw.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x0c]

th.vsxe.v v4, (a0), v12
# CHECK-INST: th.vsxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x0e]

th.vsxe.v v4, 0(a0), v12
# CHECK-INST: th.vsxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x0e]

th.vsxe.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxe.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x0c]

th.vsuxb.v v4, (a0), v12
# CHECK-INST: th.vsuxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x1e]

th.vsuxb.v v4, 0(a0), v12
# CHECK-INST: th.vsuxb.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x1e]

th.vsuxb.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsuxb.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x1c]

th.vsuxh.v v4, (a0), v12
# CHECK-INST: th.vsuxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x1e]

th.vsuxh.v v4, 0(a0), v12
# CHECK-INST: th.vsuxh.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x1e]

th.vsuxh.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsuxh.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x1c]

th.vsuxw.v v4, (a0), v12
# CHECK-INST: th.vsuxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x1e]

th.vsuxw.v v4, 0(a0), v12
# CHECK-INST: th.vsuxw.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x1e]

th.vsuxw.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsuxw.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x1c]

th.vsuxe.v v4, (a0), v12
# CHECK-INST: th.vsuxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x1e]

th.vsuxe.v v4, 0(a0), v12
# CHECK-INST: th.vsuxe.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x1e]

th.vsuxe.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsuxe.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x1c]

th.vlbff.v v4, (a0)
# CHECK-INST: th.vlbff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x13]

th.vlbff.v v4, 0(a0)
# CHECK-INST: th.vlbff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x13]

th.vlbff.v v4, (a0), v0.t
# CHECK-INST: th.vlbff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x11]

th.vlhff.v v4, (a0)
# CHECK-INST: th.vlhff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x13]

th.vlhff.v v4, 0(a0)
# CHECK-INST: th.vlhff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x13]

th.vlhff.v v4, (a0), v0.t
# CHECK-INST: th.vlhff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x11]

th.vlwff.v v4, (a0)
# CHECK-INST: th.vlwff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x13]

th.vlwff.v v4, 0(a0)
# CHECK-INST: th.vlwff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x13]

th.vlwff.v v4, (a0), v0.t
# CHECK-INST: th.vlwff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x11]

th.vlbuff.v v4, (a0)
# CHECK-INST: th.vlbuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x03]

th.vlbuff.v v4, 0(a0)
# CHECK-INST: th.vlbuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x03]

th.vlbuff.v v4, (a0), v0.t
# CHECK-INST: th.vlbuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x01]

th.vlhuff.v v4, (a0)
# CHECK-INST: th.vlhuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x03]

th.vlhuff.v v4, 0(a0)
# CHECK-INST: th.vlhuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x03]

th.vlhuff.v v4, (a0), v0.t
# CHECK-INST: th.vlhuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x01]

th.vlwuff.v v4, (a0)
# CHECK-INST: th.vlwuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x03]

th.vlwuff.v v4, 0(a0)
# CHECK-INST: th.vlwuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x03]

th.vlwuff.v v4, (a0), v0.t
# CHECK-INST: th.vlwuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x01]

th.vleff.v v4, (a0)
# CHECK-INST: th.vleff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x03]

th.vleff.v v4, 0(a0)
# CHECK-INST: th.vleff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x03]

th.vleff.v v4, (a0), v0.t
# CHECK-INST: th.vleff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x01]

th.vlseg2b.v v4, (a0)
# CHECK-INST: th.vlseg2b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x32]

th.vlseg2b.v v4, 0(a0)
# CHECK-INST: th.vlseg2b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x32]

th.vlseg2b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x30]

th.vlseg2h.v v4, (a0)
# CHECK-INST: th.vlseg2h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x32]

th.vlseg2h.v v4, 0(a0)
# CHECK-INST: th.vlseg2h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x32]

th.vlseg2h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x30]

th.vlseg2w.v v4, (a0)
# CHECK-INST: th.vlseg2w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x32]

th.vlseg2w.v v4, 0(a0)
# CHECK-INST: th.vlseg2w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x32]

th.vlseg2w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x30]

th.vlseg2bu.v v4, (a0)
# CHECK-INST: th.vlseg2bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x22]

th.vlseg2bu.v v4, 0(a0)
# CHECK-INST: th.vlseg2bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x22]

th.vlseg2bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x20]

th.vlseg2hu.v v4, (a0)
# CHECK-INST: th.vlseg2hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x22]

th.vlseg2hu.v v4, 0(a0)
# CHECK-INST: th.vlseg2hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x22]

th.vlseg2hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x20]

th.vlseg2wu.v v4, (a0)
# CHECK-INST: th.vlseg2wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x22]

th.vlseg2wu.v v4, 0(a0)
# CHECK-INST: th.vlseg2wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x22]

th.vlseg2wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x20]

th.vlseg2e.v v4, (a0)
# CHECK-INST: th.vlseg2e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x22]

th.vlseg2e.v v4, 0(a0)
# CHECK-INST: th.vlseg2e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x22]

th.vlseg2e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x20]

th.vsseg2b.v v4, (a0)
# CHECK-INST: th.vsseg2b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x22]

th.vsseg2b.v v4, 0(a0)
# CHECK-INST: th.vsseg2b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x22]

th.vsseg2b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg2b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0x20]

th.vsseg2h.v v4, (a0)
# CHECK-INST: th.vsseg2h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x22]

th.vsseg2h.v v4, 0(a0)
# CHECK-INST: th.vsseg2h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x22]

th.vsseg2h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg2h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0x20]

th.vsseg2w.v v4, (a0)
# CHECK-INST: th.vsseg2w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x22]

th.vsseg2w.v v4, 0(a0)
# CHECK-INST: th.vsseg2w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x22]

th.vsseg2w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg2w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0x20]

th.vsseg2e.v v4, (a0)
# CHECK-INST: th.vsseg2e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x22]

th.vsseg2e.v v4, 0(a0)
# CHECK-INST: th.vsseg2e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x22]

th.vsseg2e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg2e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0x20]

th.vlseg3b.v v4, (a0)
# CHECK-INST: th.vlseg3b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x52]

th.vlseg3b.v v4, 0(a0)
# CHECK-INST: th.vlseg3b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x52]

th.vlseg3b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x50]

th.vlseg3h.v v4, (a0)
# CHECK-INST: th.vlseg3h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x52]

th.vlseg3h.v v4, 0(a0)
# CHECK-INST: th.vlseg3h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x52]

th.vlseg3h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x50]

th.vlseg3w.v v4, (a0)
# CHECK-INST: th.vlseg3w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x52]

th.vlseg3w.v v4, 0(a0)
# CHECK-INST: th.vlseg3w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x52]

th.vlseg3w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x50]

th.vlseg3bu.v v4, (a0)
# CHECK-INST: th.vlseg3bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x42]

th.vlseg3bu.v v4, 0(a0)
# CHECK-INST: th.vlseg3bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x42]

th.vlseg3bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x40]

th.vlseg3hu.v v4, (a0)
# CHECK-INST: th.vlseg3hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x42]

th.vlseg3hu.v v4, 0(a0)
# CHECK-INST: th.vlseg3hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x42]

th.vlseg3hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x40]

th.vlseg3wu.v v4, (a0)
# CHECK-INST: th.vlseg3wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x42]

th.vlseg3wu.v v4, 0(a0)
# CHECK-INST: th.vlseg3wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x42]

th.vlseg3wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x40]

th.vlseg3e.v v4, (a0)
# CHECK-INST: th.vlseg3e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x42]

th.vlseg3e.v v4, 0(a0)
# CHECK-INST: th.vlseg3e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x42]

th.vlseg3e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x40]

th.vsseg3b.v v4, (a0)
# CHECK-INST: th.vsseg3b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x42]

th.vsseg3b.v v4, 0(a0)
# CHECK-INST: th.vsseg3b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x42]

th.vsseg3b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg3b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0x40]

th.vsseg3h.v v4, (a0)
# CHECK-INST: th.vsseg3h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x42]

th.vsseg3h.v v4, 0(a0)
# CHECK-INST: th.vsseg3h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x42]

th.vsseg3h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg3h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0x40]

th.vsseg3w.v v4, (a0)
# CHECK-INST: th.vsseg3w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x42]

th.vsseg3w.v v4, 0(a0)
# CHECK-INST: th.vsseg3w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x42]

th.vsseg3w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg3w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0x40]

th.vsseg3e.v v4, (a0)
# CHECK-INST: th.vsseg3e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x42]

th.vsseg3e.v v4, 0(a0)
# CHECK-INST: th.vsseg3e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x42]

th.vsseg3e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg3e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0x40]

th.vlseg4b.v v4, (a0)
# CHECK-INST: th.vlseg4b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x72]

th.vlseg4b.v v4, 0(a0)
# CHECK-INST: th.vlseg4b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x72]

th.vlseg4b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x70]

th.vlseg4h.v v4, (a0)
# CHECK-INST: th.vlseg4h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x72]

th.vlseg4h.v v4, 0(a0)
# CHECK-INST: th.vlseg4h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x72]

th.vlseg4h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x70]

th.vlseg4w.v v4, (a0)
# CHECK-INST: th.vlseg4w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x72]

th.vlseg4w.v v4, 0(a0)
# CHECK-INST: th.vlseg4w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x72]

th.vlseg4w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x70]

th.vlseg4bu.v v4, (a0)
# CHECK-INST: th.vlseg4bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x62]

th.vlseg4bu.v v4, 0(a0)
# CHECK-INST: th.vlseg4bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x62]

th.vlseg4bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x60]

th.vlseg4hu.v v4, (a0)
# CHECK-INST: th.vlseg4hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x62]

th.vlseg4hu.v v4, 0(a0)
# CHECK-INST: th.vlseg4hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x62]

th.vlseg4hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x60]

th.vlseg4wu.v v4, (a0)
# CHECK-INST: th.vlseg4wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x62]

th.vlseg4wu.v v4, 0(a0)
# CHECK-INST: th.vlseg4wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x62]

th.vlseg4wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x60]

th.vlseg4e.v v4, (a0)
# CHECK-INST: th.vlseg4e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x62]

th.vlseg4e.v v4, 0(a0)
# CHECK-INST: th.vlseg4e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x62]

th.vlseg4e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x60]

th.vsseg4b.v v4, (a0)
# CHECK-INST: th.vsseg4b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x62]

th.vsseg4b.v v4, 0(a0)
# CHECK-INST: th.vsseg4b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x62]

th.vsseg4b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg4b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0x60]

th.vsseg4h.v v4, (a0)
# CHECK-INST: th.vsseg4h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x62]

th.vsseg4h.v v4, 0(a0)
# CHECK-INST: th.vsseg4h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x62]

th.vsseg4h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg4h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0x60]

th.vsseg4w.v v4, (a0)
# CHECK-INST: th.vsseg4w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x62]

th.vsseg4w.v v4, 0(a0)
# CHECK-INST: th.vsseg4w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x62]

th.vsseg4w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg4w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0x60]

th.vsseg4e.v v4, (a0)
# CHECK-INST: th.vsseg4e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x62]

th.vsseg4e.v v4, 0(a0)
# CHECK-INST: th.vsseg4e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x62]

th.vsseg4e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg4e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0x60]

th.vlseg5b.v v4, (a0)
# CHECK-INST: th.vlseg5b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x92]

th.vlseg5b.v v4, 0(a0)
# CHECK-INST: th.vlseg5b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x92]

th.vlseg5b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x90]

th.vlseg5h.v v4, (a0)
# CHECK-INST: th.vlseg5h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x92]

th.vlseg5h.v v4, 0(a0)
# CHECK-INST: th.vlseg5h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x92]

th.vlseg5h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x90]

th.vlseg5w.v v4, (a0)
# CHECK-INST: th.vlseg5w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x92]

th.vlseg5w.v v4, 0(a0)
# CHECK-INST: th.vlseg5w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x92]

th.vlseg5w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x90]

th.vlseg5bu.v v4, (a0)
# CHECK-INST: th.vlseg5bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x82]

th.vlseg5bu.v v4, 0(a0)
# CHECK-INST: th.vlseg5bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x82]

th.vlseg5bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x80]

th.vlseg5hu.v v4, (a0)
# CHECK-INST: th.vlseg5hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x82]

th.vlseg5hu.v v4, 0(a0)
# CHECK-INST: th.vlseg5hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x82]

th.vlseg5hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x80]

th.vlseg5wu.v v4, (a0)
# CHECK-INST: th.vlseg5wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x82]

th.vlseg5wu.v v4, 0(a0)
# CHECK-INST: th.vlseg5wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x82]

th.vlseg5wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x80]

th.vlseg5e.v v4, (a0)
# CHECK-INST: th.vlseg5e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x82]

th.vlseg5e.v v4, 0(a0)
# CHECK-INST: th.vlseg5e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x82]

th.vlseg5e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x80]

th.vsseg5b.v v4, (a0)
# CHECK-INST: th.vsseg5b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x82]

th.vsseg5b.v v4, 0(a0)
# CHECK-INST: th.vsseg5b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0x82]

th.vsseg5b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg5b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0x80]

th.vsseg5h.v v4, (a0)
# CHECK-INST: th.vsseg5h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x82]

th.vsseg5h.v v4, 0(a0)
# CHECK-INST: th.vsseg5h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0x82]

th.vsseg5h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg5h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0x80]

th.vsseg5w.v v4, (a0)
# CHECK-INST: th.vsseg5w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x82]

th.vsseg5w.v v4, 0(a0)
# CHECK-INST: th.vsseg5w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0x82]

th.vsseg5w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg5w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0x80]

th.vsseg5e.v v4, (a0)
# CHECK-INST: th.vsseg5e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x82]

th.vsseg5e.v v4, 0(a0)
# CHECK-INST: th.vsseg5e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0x82]

th.vsseg5e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg5e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0x80]

th.vlseg6b.v v4, (a0)
# CHECK-INST: th.vlseg6b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xb2]

th.vlseg6b.v v4, 0(a0)
# CHECK-INST: th.vlseg6b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xb2]

th.vlseg6b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xb0]

th.vlseg6h.v v4, (a0)
# CHECK-INST: th.vlseg6h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xb2]

th.vlseg6h.v v4, 0(a0)
# CHECK-INST: th.vlseg6h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xb2]

th.vlseg6h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xb0]

th.vlseg6w.v v4, (a0)
# CHECK-INST: th.vlseg6w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xb2]

th.vlseg6w.v v4, 0(a0)
# CHECK-INST: th.vlseg6w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xb2]

th.vlseg6w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xb0]

th.vlseg6bu.v v4, (a0)
# CHECK-INST: th.vlseg6bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xa2]

th.vlseg6bu.v v4, 0(a0)
# CHECK-INST: th.vlseg6bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xa2]

th.vlseg6bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xa0]

th.vlseg6hu.v v4, (a0)
# CHECK-INST: th.vlseg6hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xa2]

th.vlseg6hu.v v4, 0(a0)
# CHECK-INST: th.vlseg6hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xa2]

th.vlseg6hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xa0]

th.vlseg6wu.v v4, (a0)
# CHECK-INST: th.vlseg6wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xa2]

th.vlseg6wu.v v4, 0(a0)
# CHECK-INST: th.vlseg6wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xa2]

th.vlseg6wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xa0]

th.vlseg6e.v v4, (a0)
# CHECK-INST: th.vlseg6e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xa2]

th.vlseg6e.v v4, 0(a0)
# CHECK-INST: th.vlseg6e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xa2]

th.vlseg6e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xa0]

th.vsseg6b.v v4, (a0)
# CHECK-INST: th.vsseg6b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xa2]

th.vsseg6b.v v4, 0(a0)
# CHECK-INST: th.vsseg6b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xa2]

th.vsseg6b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg6b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0xa0]

th.vsseg6h.v v4, (a0)
# CHECK-INST: th.vsseg6h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xa2]

th.vsseg6h.v v4, 0(a0)
# CHECK-INST: th.vsseg6h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xa2]

th.vsseg6h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg6h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0xa0]

th.vsseg6w.v v4, (a0)
# CHECK-INST: th.vsseg6w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xa2]

th.vsseg6w.v v4, 0(a0)
# CHECK-INST: th.vsseg6w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xa2]

th.vsseg6w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg6w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0xa0]

th.vsseg6e.v v4, (a0)
# CHECK-INST: th.vsseg6e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xa2]

th.vsseg6e.v v4, 0(a0)
# CHECK-INST: th.vsseg6e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xa2]

th.vsseg6e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg6e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0xa0]

th.vlseg7b.v v4, (a0)
# CHECK-INST: th.vlseg7b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xd2]

th.vlseg7b.v v4, 0(a0)
# CHECK-INST: th.vlseg7b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xd2]

th.vlseg7b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xd0]

th.vlseg7h.v v4, (a0)
# CHECK-INST: th.vlseg7h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xd2]

th.vlseg7h.v v4, 0(a0)
# CHECK-INST: th.vlseg7h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xd2]

th.vlseg7h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xd0]

th.vlseg7w.v v4, (a0)
# CHECK-INST: th.vlseg7w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xd2]

th.vlseg7w.v v4, 0(a0)
# CHECK-INST: th.vlseg7w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xd2]

th.vlseg7w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xd0]

th.vlseg7bu.v v4, (a0)
# CHECK-INST: th.vlseg7bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xc2]

th.vlseg7bu.v v4, 0(a0)
# CHECK-INST: th.vlseg7bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xc2]

th.vlseg7bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xc0]

th.vlseg7hu.v v4, (a0)
# CHECK-INST: th.vlseg7hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xc2]

th.vlseg7hu.v v4, 0(a0)
# CHECK-INST: th.vlseg7hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xc2]

th.vlseg7hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xc0]

th.vlseg7wu.v v4, (a0)
# CHECK-INST: th.vlseg7wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xc2]

th.vlseg7wu.v v4, 0(a0)
# CHECK-INST: th.vlseg7wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xc2]

th.vlseg7wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xc0]

th.vlseg7e.v v4, (a0)
# CHECK-INST: th.vlseg7e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xc2]

th.vlseg7e.v v4, 0(a0)
# CHECK-INST: th.vlseg7e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xc2]

th.vlseg7e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xc0]

th.vsseg7b.v v4, (a0)
# CHECK-INST: th.vsseg7b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xc2]

th.vsseg7b.v v4, 0(a0)
# CHECK-INST: th.vsseg7b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xc2]

th.vsseg7b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg7b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0xc0]

th.vsseg7h.v v4, (a0)
# CHECK-INST: th.vsseg7h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xc2]

th.vsseg7h.v v4, 0(a0)
# CHECK-INST: th.vsseg7h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xc2]

th.vsseg7h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg7h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0xc0]

th.vsseg7w.v v4, (a0)
# CHECK-INST: th.vsseg7w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xc2]

th.vsseg7w.v v4, 0(a0)
# CHECK-INST: th.vsseg7w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xc2]

th.vsseg7w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg7w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0xc0]

th.vsseg7e.v v4, (a0)
# CHECK-INST: th.vsseg7e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xc2]

th.vsseg7e.v v4, 0(a0)
# CHECK-INST: th.vsseg7e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xc2]

th.vsseg7e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg7e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0xc0]

th.vlseg8b.v v4, (a0)
# CHECK-INST: th.vlseg8b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xf2]

th.vlseg8b.v v4, 0(a0)
# CHECK-INST: th.vlseg8b.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xf2]

th.vlseg8b.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xf0]

th.vlseg8h.v v4, (a0)
# CHECK-INST: th.vlseg8h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xf2]

th.vlseg8h.v v4, 0(a0)
# CHECK-INST: th.vlseg8h.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xf2]

th.vlseg8h.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xf0]

th.vlseg8w.v v4, (a0)
# CHECK-INST: th.vlseg8w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xf2]

th.vlseg8w.v v4, 0(a0)
# CHECK-INST: th.vlseg8w.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xf2]

th.vlseg8w.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xf0]

th.vlseg8bu.v v4, (a0)
# CHECK-INST: th.vlseg8bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xe2]

th.vlseg8bu.v v4, 0(a0)
# CHECK-INST: th.vlseg8bu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xe2]

th.vlseg8bu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8bu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xe0]

th.vlseg8hu.v v4, (a0)
# CHECK-INST: th.vlseg8hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xe2]

th.vlseg8hu.v v4, 0(a0)
# CHECK-INST: th.vlseg8hu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xe2]

th.vlseg8hu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8hu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xe0]

th.vlseg8wu.v v4, (a0)
# CHECK-INST: th.vlseg8wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xe2]

th.vlseg8wu.v v4, 0(a0)
# CHECK-INST: th.vlseg8wu.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xe2]

th.vlseg8wu.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8wu.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xe0]

th.vlseg8e.v v4, (a0)
# CHECK-INST: th.vlseg8e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xe2]

th.vlseg8e.v v4, 0(a0)
# CHECK-INST: th.vlseg8e.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xe2]

th.vlseg8e.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xe0]

th.vsseg8b.v v4, (a0)
# CHECK-INST: th.vsseg8b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xe2]

th.vsseg8b.v v4, 0(a0)
# CHECK-INST: th.vsseg8b.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x02,0x05,0xe2]

th.vsseg8b.v v4, (a0), v0.t
# CHECK-INST: th.vsseg8b.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x02,0x05,0xe0]

th.vsseg8h.v v4, (a0)
# CHECK-INST: th.vsseg8h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xe2]

th.vsseg8h.v v4, 0(a0)
# CHECK-INST: th.vsseg8h.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x52,0x05,0xe2]

th.vsseg8h.v v4, (a0), v0.t
# CHECK-INST: th.vsseg8h.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x52,0x05,0xe0]

th.vsseg8w.v v4, (a0)
# CHECK-INST: th.vsseg8w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xe2]

th.vsseg8w.v v4, 0(a0)
# CHECK-INST: th.vsseg8w.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x62,0x05,0xe2]

th.vsseg8w.v v4, (a0), v0.t
# CHECK-INST: th.vsseg8w.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x62,0x05,0xe0]

th.vsseg8e.v v4, (a0)
# CHECK-INST: th.vsseg8e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xe2]

th.vsseg8e.v v4, 0(a0)
# CHECK-INST: th.vsseg8e.v	v4, (a0)
# CHECK-ENCODING: [0x27,0x72,0x05,0xe2]

th.vsseg8e.v v4, (a0), v0.t
# CHECK-INST: th.vsseg8e.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x27,0x72,0x05,0xe0]

th.vlsseg2b.v v4, (a0), a1
# CHECK-INST: th.vlsseg2b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x3a]

th.vlsseg2b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x3a]

th.vlsseg2b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x38]

th.vlsseg2h.v v4, (a0), a1
# CHECK-INST: th.vlsseg2h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x3a]

th.vlsseg2h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x3a]

th.vlsseg2h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x38]

th.vlsseg2w.v v4, (a0), a1
# CHECK-INST: th.vlsseg2w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x3a]

th.vlsseg2w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x3a]

th.vlsseg2w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x38]

th.vlsseg2bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg2bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x2a]

th.vlsseg2bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x2a]

th.vlsseg2bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x28]

th.vlsseg2hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg2hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x2a]

th.vlsseg2hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x2a]

th.vlsseg2hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x28]

th.vlsseg2wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg2wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x2a]

th.vlsseg2wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x2a]

th.vlsseg2wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x28]

th.vlsseg2e.v v4, (a0), a1
# CHECK-INST: th.vlsseg2e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x2a]

th.vlsseg2e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg2e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x2a]

th.vlsseg2e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0x28]

th.vssseg2b.v v4, (a0), a1
# CHECK-INST: th.vssseg2b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x2a]

th.vssseg2b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg2b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x2a]

th.vssseg2b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg2b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0x28]

th.vssseg2h.v v4, (a0), a1
# CHECK-INST: th.vssseg2h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x2a]

th.vssseg2h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg2h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x2a]

th.vssseg2h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg2h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0x28]

th.vssseg2w.v v4, (a0), a1
# CHECK-INST: th.vssseg2w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x2a]

th.vssseg2w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg2w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x2a]

th.vssseg2w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg2w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0x28]

th.vssseg2e.v v4, (a0), a1
# CHECK-INST: th.vssseg2e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x2a]

th.vssseg2e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg2e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x2a]

th.vssseg2e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg2e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0x28]

th.vlsseg3b.v v4, (a0), a1
# CHECK-INST: th.vlsseg3b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x5a]

th.vlsseg3b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x5a]

th.vlsseg3b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x58]

th.vlsseg3h.v v4, (a0), a1
# CHECK-INST: th.vlsseg3h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x5a]

th.vlsseg3h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x5a]

th.vlsseg3h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x58]

th.vlsseg3w.v v4, (a0), a1
# CHECK-INST: th.vlsseg3w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x5a]

th.vlsseg3w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x5a]

th.vlsseg3w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x58]

th.vlsseg3bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg3bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x4a]

th.vlsseg3bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x4a]

th.vlsseg3bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x48]

th.vlsseg3hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg3hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x4a]

th.vlsseg3hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x4a]

th.vlsseg3hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x48]

th.vlsseg3wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg3wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x4a]

th.vlsseg3wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x4a]

th.vlsseg3wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x48]

th.vlsseg3e.v v4, (a0), a1
# CHECK-INST: th.vlsseg3e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x4a]

th.vlsseg3e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg3e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x4a]

th.vlsseg3e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0x48]

th.vssseg3b.v v4, (a0), a1
# CHECK-INST: th.vssseg3b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x4a]

th.vssseg3b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg3b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x4a]

th.vssseg3b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg3b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0x48]

th.vssseg3h.v v4, (a0), a1
# CHECK-INST: th.vssseg3h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x4a]

th.vssseg3h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg3h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x4a]

th.vssseg3h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg3h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0x48]

th.vssseg3w.v v4, (a0), a1
# CHECK-INST: th.vssseg3w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x4a]

th.vssseg3w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg3w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x4a]

th.vssseg3w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg3w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0x48]

th.vssseg3e.v v4, (a0), a1
# CHECK-INST: th.vssseg3e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x4a]

th.vssseg3e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg3e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x4a]

th.vssseg3e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg3e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0x48]

th.vlsseg4b.v v4, (a0), a1
# CHECK-INST: th.vlsseg4b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x7a]

th.vlsseg4b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x7a]

th.vlsseg4b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x78]

th.vlsseg4h.v v4, (a0), a1
# CHECK-INST: th.vlsseg4h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x7a]

th.vlsseg4h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x7a]

th.vlsseg4h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x78]

th.vlsseg4w.v v4, (a0), a1
# CHECK-INST: th.vlsseg4w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x7a]

th.vlsseg4w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x7a]

th.vlsseg4w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x78]

th.vlsseg4bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg4bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x6a]

th.vlsseg4bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x6a]

th.vlsseg4bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x68]

th.vlsseg4hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg4hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x6a]

th.vlsseg4hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x6a]

th.vlsseg4hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x68]

th.vlsseg4wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg4wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x6a]

th.vlsseg4wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x6a]

th.vlsseg4wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x68]

th.vlsseg4e.v v4, (a0), a1
# CHECK-INST: th.vlsseg4e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x6a]

th.vlsseg4e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg4e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x6a]

th.vlsseg4e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0x68]

th.vssseg4b.v v4, (a0), a1
# CHECK-INST: th.vssseg4b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x6a]

th.vssseg4b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg4b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x6a]

th.vssseg4b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg4b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0x68]

th.vssseg4h.v v4, (a0), a1
# CHECK-INST: th.vssseg4h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x6a]

th.vssseg4h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg4h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x6a]

th.vssseg4h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg4h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0x68]

th.vssseg4w.v v4, (a0), a1
# CHECK-INST: th.vssseg4w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x6a]

th.vssseg4w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg4w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x6a]

th.vssseg4w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg4w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0x68]

th.vssseg4e.v v4, (a0), a1
# CHECK-INST: th.vssseg4e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x6a]

th.vssseg4e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg4e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x6a]

th.vssseg4e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg4e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0x68]

th.vlsseg5b.v v4, (a0), a1
# CHECK-INST: th.vlsseg5b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x9a]

th.vlsseg5b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x9a]

th.vlsseg5b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x98]

th.vlsseg5h.v v4, (a0), a1
# CHECK-INST: th.vlsseg5h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x9a]

th.vlsseg5h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x9a]

th.vlsseg5h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x98]

th.vlsseg5w.v v4, (a0), a1
# CHECK-INST: th.vlsseg5w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x9a]

th.vlsseg5w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x9a]

th.vlsseg5w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x98]

th.vlsseg5bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg5bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x8a]

th.vlsseg5bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0x8a]

th.vlsseg5bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0x88]

th.vlsseg5hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg5hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x8a]

th.vlsseg5hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0x8a]

th.vlsseg5hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0x88]

th.vlsseg5wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg5wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x8a]

th.vlsseg5wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0x8a]

th.vlsseg5wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0x88]

th.vlsseg5e.v v4, (a0), a1
# CHECK-INST: th.vlsseg5e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x8a]

th.vlsseg5e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg5e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0x8a]

th.vlsseg5e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0x88]

th.vssseg5b.v v4, (a0), a1
# CHECK-INST: th.vssseg5b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x8a]

th.vssseg5b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg5b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0x8a]

th.vssseg5b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg5b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0x88]

th.vssseg5h.v v4, (a0), a1
# CHECK-INST: th.vssseg5h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x8a]

th.vssseg5h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg5h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0x8a]

th.vssseg5h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg5h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0x88]

th.vssseg5w.v v4, (a0), a1
# CHECK-INST: th.vssseg5w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x8a]

th.vssseg5w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg5w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0x8a]

th.vssseg5w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg5w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0x88]

th.vssseg5e.v v4, (a0), a1
# CHECK-INST: th.vssseg5e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x8a]

th.vssseg5e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg5e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0x8a]

th.vssseg5e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg5e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0x88]

th.vlsseg6b.v v4, (a0), a1
# CHECK-INST: th.vlsseg6b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xba]

th.vlsseg6b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xba]

th.vlsseg6b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xb8]

th.vlsseg6h.v v4, (a0), a1
# CHECK-INST: th.vlsseg6h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xba]

th.vlsseg6h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xba]

th.vlsseg6h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xb8]

th.vlsseg6w.v v4, (a0), a1
# CHECK-INST: th.vlsseg6w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xba]

th.vlsseg6w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xba]

th.vlsseg6w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xb8]

th.vlsseg6bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg6bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xaa]

th.vlsseg6bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xaa]

th.vlsseg6bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xa8]

th.vlsseg6hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg6hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xaa]

th.vlsseg6hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xaa]

th.vlsseg6hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xa8]

th.vlsseg6wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg6wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xaa]

th.vlsseg6wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xaa]

th.vlsseg6wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xa8]

th.vlsseg6e.v v4, (a0), a1
# CHECK-INST: th.vlsseg6e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xaa]

th.vlsseg6e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg6e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xaa]

th.vlsseg6e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0xa8]

th.vssseg6b.v v4, (a0), a1
# CHECK-INST: th.vssseg6b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xaa]

th.vssseg6b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg6b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xaa]

th.vssseg6b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg6b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0xa8]

th.vssseg6h.v v4, (a0), a1
# CHECK-INST: th.vssseg6h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xaa]

th.vssseg6h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg6h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xaa]

th.vssseg6h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg6h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0xa8]

th.vssseg6w.v v4, (a0), a1
# CHECK-INST: th.vssseg6w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xaa]

th.vssseg6w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg6w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xaa]

th.vssseg6w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg6w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0xa8]

th.vssseg6e.v v4, (a0), a1
# CHECK-INST: th.vssseg6e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xaa]

th.vssseg6e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg6e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xaa]

th.vssseg6e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg6e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0xa8]

th.vlsseg7b.v v4, (a0), a1
# CHECK-INST: th.vlsseg7b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xda]

th.vlsseg7b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xda]

th.vlsseg7b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xd8]

th.vlsseg7h.v v4, (a0), a1
# CHECK-INST: th.vlsseg7h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xda]

th.vlsseg7h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xda]

th.vlsseg7h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xd8]

th.vlsseg7w.v v4, (a0), a1
# CHECK-INST: th.vlsseg7w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xda]

th.vlsseg7w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xda]

th.vlsseg7w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xd8]

th.vlsseg7bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg7bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xca]

th.vlsseg7bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xca]

th.vlsseg7bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xc8]

th.vlsseg7hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg7hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xca]

th.vlsseg7hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xca]

th.vlsseg7hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xc8]

th.vlsseg7wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg7wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xca]

th.vlsseg7wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xca]

th.vlsseg7wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xc8]

th.vlsseg7e.v v4, (a0), a1
# CHECK-INST: th.vlsseg7e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xca]

th.vlsseg7e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg7e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xca]

th.vlsseg7e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0xc8]

th.vssseg7b.v v4, (a0), a1
# CHECK-INST: th.vssseg7b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xca]

th.vssseg7b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg7b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xca]

th.vssseg7b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg7b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0xc8]

th.vssseg7h.v v4, (a0), a1
# CHECK-INST: th.vssseg7h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xca]

th.vssseg7h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg7h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xca]

th.vssseg7h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg7h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0xc8]

th.vssseg7w.v v4, (a0), a1
# CHECK-INST: th.vssseg7w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xca]

th.vssseg7w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg7w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xca]

th.vssseg7w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg7w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0xc8]

th.vssseg7e.v v4, (a0), a1
# CHECK-INST: th.vssseg7e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xca]

th.vssseg7e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg7e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xca]

th.vssseg7e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg7e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0xc8]

th.vlsseg8b.v v4, (a0), a1
# CHECK-INST: th.vlsseg8b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xfa]

th.vlsseg8b.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8b.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xfa]

th.vlsseg8b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xf8]

th.vlsseg8h.v v4, (a0), a1
# CHECK-INST: th.vlsseg8h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xfa]

th.vlsseg8h.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8h.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xfa]

th.vlsseg8h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xf8]

th.vlsseg8w.v v4, (a0), a1
# CHECK-INST: th.vlsseg8w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xfa]

th.vlsseg8w.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8w.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xfa]

th.vlsseg8w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xf8]

th.vlsseg8bu.v v4, (a0), a1
# CHECK-INST: th.vlsseg8bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xea]

th.vlsseg8bu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8bu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x02,0xb5,0xea]

th.vlsseg8bu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8bu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x02,0xb5,0xe8]

th.vlsseg8hu.v v4, (a0), a1
# CHECK-INST: th.vlsseg8hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xea]

th.vlsseg8hu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8hu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x52,0xb5,0xea]

th.vlsseg8hu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8hu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x52,0xb5,0xe8]

th.vlsseg8wu.v v4, (a0), a1
# CHECK-INST: th.vlsseg8wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xea]

th.vlsseg8wu.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8wu.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x62,0xb5,0xea]

th.vlsseg8wu.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8wu.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x62,0xb5,0xe8]

th.vlsseg8e.v v4, (a0), a1
# CHECK-INST: th.vlsseg8e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xea]

th.vlsseg8e.v v4, 0(a0), a1
# CHECK-INST: th.vlsseg8e.v	v4, (a0), a1
# CHECK-ENCODING: [0x07,0x72,0xb5,0xea]

th.vlsseg8e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x72,0xb5,0xe8]

th.vssseg8b.v v4, (a0), a1
# CHECK-INST: th.vssseg8b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xea]

th.vssseg8b.v v4, 0(a0), a1
# CHECK-INST: th.vssseg8b.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x02,0xb5,0xea]

th.vssseg8b.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg8b.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x02,0xb5,0xe8]

th.vssseg8h.v v4, (a0), a1
# CHECK-INST: th.vssseg8h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xea]

th.vssseg8h.v v4, 0(a0), a1
# CHECK-INST: th.vssseg8h.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x52,0xb5,0xea]

th.vssseg8h.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg8h.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x52,0xb5,0xe8]

th.vssseg8w.v v4, (a0), a1
# CHECK-INST: th.vssseg8w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xea]

th.vssseg8w.v v4, 0(a0), a1
# CHECK-INST: th.vssseg8w.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x62,0xb5,0xea]

th.vssseg8w.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg8w.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x62,0xb5,0xe8]

th.vssseg8e.v v4, (a0), a1
# CHECK-INST: th.vssseg8e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xea]

th.vssseg8e.v v4, 0(a0), a1
# CHECK-INST: th.vssseg8e.v	v4, (a0), a1
# CHECK-ENCODING: [0x27,0x72,0xb5,0xea]

th.vssseg8e.v v4, (a0), a1, v0.t
# CHECK-INST: th.vssseg8e.v	v4, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x72,0xb5,0xe8]

th.vlxseg2b.v v4, (a0), v12
# CHECK-INST: th.vlxseg2b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x3e]

th.vlxseg2b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x3e]

th.vlxseg2b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x3c]

th.vlxseg2h.v v4, (a0), v12
# CHECK-INST: th.vlxseg2h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x3e]

th.vlxseg2h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x3e]

th.vlxseg2h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x3c]

th.vlxseg2w.v v4, (a0), v12
# CHECK-INST: th.vlxseg2w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x3e]

th.vlxseg2w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x3e]

th.vlxseg2w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x3c]

th.vlxseg2bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg2bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x2e]

th.vlxseg2bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x2e]

th.vlxseg2bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x2c]

th.vlxseg2hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg2hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x2e]

th.vlxseg2hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x2e]

th.vlxseg2hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x2c]

th.vlxseg2wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg2wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x2e]

th.vlxseg2wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x2e]

th.vlxseg2wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x2c]

th.vlxseg2e.v v4, (a0), v12
# CHECK-INST: th.vlxseg2e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x2e]

th.vlxseg2e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg2e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x2e]

th.vlxseg2e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg2e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0x2c]

th.vsxseg2b.v v4, (a0), v12
# CHECK-INST: th.vsxseg2b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x2e]

th.vsxseg2b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg2b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x2e]

th.vsxseg2b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg2b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x2c]

th.vsxseg2h.v v4, (a0), v12
# CHECK-INST: th.vsxseg2h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x2e]

th.vsxseg2h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg2h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x2e]

th.vsxseg2h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg2h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x2c]

th.vsxseg2w.v v4, (a0), v12
# CHECK-INST: th.vsxseg2w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x2e]

th.vsxseg2w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg2w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x2e]

th.vsxseg2w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg2w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x2c]

th.vsxseg2e.v v4, (a0), v12
# CHECK-INST: th.vsxseg2e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x2e]

th.vsxseg2e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg2e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x2e]

th.vsxseg2e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg2e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x2c]

th.vlxseg3b.v v4, (a0), v12
# CHECK-INST: th.vlxseg3b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x5e]

th.vlxseg3b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x5e]

th.vlxseg3b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x5c]

th.vlxseg3h.v v4, (a0), v12
# CHECK-INST: th.vlxseg3h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x5e]

th.vlxseg3h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x5e]

th.vlxseg3h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x5c]

th.vlxseg3w.v v4, (a0), v12
# CHECK-INST: th.vlxseg3w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x5e]

th.vlxseg3w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x5e]

th.vlxseg3w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x5c]

th.vlxseg3bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg3bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x4e]

th.vlxseg3bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x4e]

th.vlxseg3bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x4c]

th.vlxseg3hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg3hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x4e]

th.vlxseg3hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x4e]

th.vlxseg3hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x4c]

th.vlxseg3wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg3wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x4e]

th.vlxseg3wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x4e]

th.vlxseg3wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x4c]

th.vlxseg3e.v v4, (a0), v12
# CHECK-INST: th.vlxseg3e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x4e]

th.vlxseg3e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg3e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x4e]

th.vlxseg3e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg3e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0x4c]

th.vsxseg3b.v v4, (a0), v12
# CHECK-INST: th.vsxseg3b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x4e]

th.vsxseg3b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg3b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x4e]

th.vsxseg3b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg3b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x4c]

th.vsxseg3h.v v4, (a0), v12
# CHECK-INST: th.vsxseg3h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x4e]

th.vsxseg3h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg3h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x4e]

th.vsxseg3h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg3h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x4c]

th.vsxseg3w.v v4, (a0), v12
# CHECK-INST: th.vsxseg3w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x4e]

th.vsxseg3w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg3w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x4e]

th.vsxseg3w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg3w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x4c]

th.vsxseg3e.v v4, (a0), v12
# CHECK-INST: th.vsxseg3e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x4e]

th.vsxseg3e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg3e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x4e]

th.vsxseg3e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg3e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x4c]

th.vlxseg4b.v v4, (a0), v12
# CHECK-INST: th.vlxseg4b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x7e]

th.vlxseg4b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x7e]

th.vlxseg4b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x7c]

th.vlxseg4h.v v4, (a0), v12
# CHECK-INST: th.vlxseg4h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x7e]

th.vlxseg4h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x7e]

th.vlxseg4h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x7c]

th.vlxseg4w.v v4, (a0), v12
# CHECK-INST: th.vlxseg4w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x7e]

th.vlxseg4w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x7e]

th.vlxseg4w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x7c]

th.vlxseg4bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg4bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x6e]

th.vlxseg4bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x6e]

th.vlxseg4bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x6c]

th.vlxseg4hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg4hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x6e]

th.vlxseg4hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x6e]

th.vlxseg4hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x6c]

th.vlxseg4wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg4wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x6e]

th.vlxseg4wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x6e]

th.vlxseg4wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x6c]

th.vlxseg4e.v v4, (a0), v12
# CHECK-INST: th.vlxseg4e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x6e]

th.vlxseg4e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg4e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x6e]

th.vlxseg4e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg4e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0x6c]

th.vsxseg4b.v v4, (a0), v12
# CHECK-INST: th.vsxseg4b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x6e]

th.vsxseg4b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg4b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x6e]

th.vsxseg4b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg4b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x6c]

th.vsxseg4h.v v4, (a0), v12
# CHECK-INST: th.vsxseg4h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x6e]

th.vsxseg4h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg4h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x6e]

th.vsxseg4h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg4h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x6c]

th.vsxseg4w.v v4, (a0), v12
# CHECK-INST: th.vsxseg4w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x6e]

th.vsxseg4w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg4w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x6e]

th.vsxseg4w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg4w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x6c]

th.vsxseg4e.v v4, (a0), v12
# CHECK-INST: th.vsxseg4e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x6e]

th.vsxseg4e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg4e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x6e]

th.vsxseg4e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg4e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x6c]

th.vlxseg5b.v v4, (a0), v12
# CHECK-INST: th.vlxseg5b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x9e]

th.vlxseg5b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x9e]

th.vlxseg5b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x9c]

th.vlxseg5h.v v4, (a0), v12
# CHECK-INST: th.vlxseg5h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x9e]

th.vlxseg5h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x9e]

th.vlxseg5h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x9c]

th.vlxseg5w.v v4, (a0), v12
# CHECK-INST: th.vlxseg5w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x9e]

th.vlxseg5w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x9e]

th.vlxseg5w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x9c]

th.vlxseg5bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg5bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x8e]

th.vlxseg5bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0x8e]

th.vlxseg5bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0x8c]

th.vlxseg5hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg5hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x8e]

th.vlxseg5hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0x8e]

th.vlxseg5hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0x8c]

th.vlxseg5wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg5wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x8e]

th.vlxseg5wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0x8e]

th.vlxseg5wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0x8c]

th.vlxseg5e.v v4, (a0), v12
# CHECK-INST: th.vlxseg5e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x8e]

th.vlxseg5e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg5e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0x8e]

th.vlxseg5e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg5e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0x8c]

th.vsxseg5b.v v4, (a0), v12
# CHECK-INST: th.vsxseg5b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x8e]

th.vsxseg5b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg5b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0x8e]

th.vsxseg5b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg5b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0x8c]

th.vsxseg5h.v v4, (a0), v12
# CHECK-INST: th.vsxseg5h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x8e]

th.vsxseg5h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg5h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0x8e]

th.vsxseg5h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg5h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0x8c]

th.vsxseg5w.v v4, (a0), v12
# CHECK-INST: th.vsxseg5w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x8e]

th.vsxseg5w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg5w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0x8e]

th.vsxseg5w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg5w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0x8c]

th.vsxseg5e.v v4, (a0), v12
# CHECK-INST: th.vsxseg5e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x8e]

th.vsxseg5e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg5e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0x8e]

th.vsxseg5e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg5e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0x8c]

th.vlxseg6b.v v4, (a0), v12
# CHECK-INST: th.vlxseg6b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xbe]

th.vlxseg6b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xbe]

th.vlxseg6b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xbc]

th.vlxseg6h.v v4, (a0), v12
# CHECK-INST: th.vlxseg6h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xbe]

th.vlxseg6h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xbe]

th.vlxseg6h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xbc]

th.vlxseg6w.v v4, (a0), v12
# CHECK-INST: th.vlxseg6w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xbe]

th.vlxseg6w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xbe]

th.vlxseg6w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xbc]

th.vlxseg6bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg6bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xae]

th.vlxseg6bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xae]

th.vlxseg6bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xac]

th.vlxseg6hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg6hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xae]

th.vlxseg6hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xae]

th.vlxseg6hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xac]

th.vlxseg6wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg6wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xae]

th.vlxseg6wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xae]

th.vlxseg6wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xac]

th.vlxseg6e.v v4, (a0), v12
# CHECK-INST: th.vlxseg6e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xae]

th.vlxseg6e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg6e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xae]

th.vlxseg6e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg6e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0xac]

th.vsxseg6b.v v4, (a0), v12
# CHECK-INST: th.vsxseg6b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xae]

th.vsxseg6b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg6b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xae]

th.vsxseg6b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg6b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0xac]

th.vsxseg6h.v v4, (a0), v12
# CHECK-INST: th.vsxseg6h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xae]

th.vsxseg6h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg6h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xae]

th.vsxseg6h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg6h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0xac]

th.vsxseg6w.v v4, (a0), v12
# CHECK-INST: th.vsxseg6w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xae]

th.vsxseg6w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg6w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xae]

th.vsxseg6w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg6w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0xac]

th.vsxseg6e.v v4, (a0), v12
# CHECK-INST: th.vsxseg6e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xae]

th.vsxseg6e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg6e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xae]

th.vsxseg6e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg6e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0xac]

th.vlxseg7b.v v4, (a0), v12
# CHECK-INST: th.vlxseg7b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xde]

th.vlxseg7b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xde]

th.vlxseg7b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xdc]

th.vlxseg7h.v v4, (a0), v12
# CHECK-INST: th.vlxseg7h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xde]

th.vlxseg7h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xde]

th.vlxseg7h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xdc]

th.vlxseg7w.v v4, (a0), v12
# CHECK-INST: th.vlxseg7w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xde]

th.vlxseg7w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xde]

th.vlxseg7w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xdc]

th.vlxseg7bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg7bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xce]

th.vlxseg7bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xce]

th.vlxseg7bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xcc]

th.vlxseg7hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg7hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xce]

th.vlxseg7hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xce]

th.vlxseg7hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xcc]

th.vlxseg7wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg7wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xce]

th.vlxseg7wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xce]

th.vlxseg7wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xcc]

th.vlxseg7e.v v4, (a0), v12
# CHECK-INST: th.vlxseg7e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xce]

th.vlxseg7e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg7e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xce]

th.vlxseg7e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg7e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0xcc]

th.vsxseg7b.v v4, (a0), v12
# CHECK-INST: th.vsxseg7b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xce]

th.vsxseg7b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg7b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xce]

th.vsxseg7b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg7b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0xcc]

th.vsxseg7h.v v4, (a0), v12
# CHECK-INST: th.vsxseg7h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xce]

th.vsxseg7h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg7h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xce]

th.vsxseg7h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg7h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0xcc]

th.vsxseg7w.v v4, (a0), v12
# CHECK-INST: th.vsxseg7w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xce]

th.vsxseg7w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg7w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xce]

th.vsxseg7w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg7w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0xcc]

th.vsxseg7e.v v4, (a0), v12
# CHECK-INST: th.vsxseg7e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xce]

th.vsxseg7e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg7e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xce]

th.vsxseg7e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg7e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0xcc]

th.vlxseg8b.v v4, (a0), v12
# CHECK-INST: th.vlxseg8b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xfe]

th.vlxseg8b.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8b.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xfe]

th.vlxseg8b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xfc]

th.vlxseg8h.v v4, (a0), v12
# CHECK-INST: th.vlxseg8h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xfe]

th.vlxseg8h.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8h.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xfe]

th.vlxseg8h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xfc]

th.vlxseg8w.v v4, (a0), v12
# CHECK-INST: th.vlxseg8w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xfe]

th.vlxseg8w.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8w.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xfe]

th.vlxseg8w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xfc]

th.vlxseg8bu.v v4, (a0), v12
# CHECK-INST: th.vlxseg8bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xee]

th.vlxseg8bu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8bu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x02,0xc5,0xee]

th.vlxseg8bu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8bu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x02,0xc5,0xec]

th.vlxseg8hu.v v4, (a0), v12
# CHECK-INST: th.vlxseg8hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xee]

th.vlxseg8hu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8hu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x52,0xc5,0xee]

th.vlxseg8hu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8hu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x52,0xc5,0xec]

th.vlxseg8wu.v v4, (a0), v12
# CHECK-INST: th.vlxseg8wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xee]

th.vlxseg8wu.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8wu.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x62,0xc5,0xee]

th.vlxseg8wu.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8wu.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x62,0xc5,0xec]

th.vlxseg8e.v v4, (a0), v12
# CHECK-INST: th.vlxseg8e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xee]

th.vlxseg8e.v v4, 0(a0), v12
# CHECK-INST: th.vlxseg8e.v	v4, (a0), v12
# CHECK-ENCODING: [0x07,0x72,0xc5,0xee]

th.vlxseg8e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vlxseg8e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x07,0x72,0xc5,0xec]

th.vsxseg8b.v v4, (a0), v12
# CHECK-INST: th.vsxseg8b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xee]

th.vsxseg8b.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg8b.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x02,0xc5,0xee]

th.vsxseg8b.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg8b.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x02,0xc5,0xec]

th.vsxseg8h.v v4, (a0), v12
# CHECK-INST: th.vsxseg8h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xee]

th.vsxseg8h.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg8h.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x52,0xc5,0xee]

th.vsxseg8h.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg8h.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x52,0xc5,0xec]

th.vsxseg8w.v v4, (a0), v12
# CHECK-INST: th.vsxseg8w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xee]

th.vsxseg8w.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg8w.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x62,0xc5,0xee]

th.vsxseg8w.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg8w.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x62,0xc5,0xec]

th.vsxseg8e.v v4, (a0), v12
# CHECK-INST: th.vsxseg8e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xee]

th.vsxseg8e.v v4, 0(a0), v12
# CHECK-INST: th.vsxseg8e.v	v4, (a0), v12
# CHECK-ENCODING: [0x27,0x72,0xc5,0xee]

th.vsxseg8e.v v4, (a0), v12, v0.t
# CHECK-INST: th.vsxseg8e.v	v4, (a0), v12, v0.t
# CHECK-ENCODING: [0x27,0x72,0xc5,0xec]

th.vlseg2bff.v v4, (a0)
# CHECK-INST: th.vlseg2bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x33]

th.vlseg2bff.v v4, 0(a0)
# CHECK-INST: th.vlseg2bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x33]

th.vlseg2bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x31]

th.vlseg2hff.v v4, (a0)
# CHECK-INST: th.vlseg2hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x33]

th.vlseg2hff.v v4, 0(a0)
# CHECK-INST: th.vlseg2hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x33]

th.vlseg2hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x31]

th.vlseg2wff.v v4, (a0)
# CHECK-INST: th.vlseg2wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x33]

th.vlseg2wff.v v4, 0(a0)
# CHECK-INST: th.vlseg2wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x33]

th.vlseg2wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x31]

th.vlseg2buff.v v4, (a0)
# CHECK-INST: th.vlseg2buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x23]

th.vlseg2buff.v v4, 0(a0)
# CHECK-INST: th.vlseg2buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x23]

th.vlseg2buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x21]

th.vlseg2huff.v v4, (a0)
# CHECK-INST: th.vlseg2huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x23]

th.vlseg2huff.v v4, 0(a0)
# CHECK-INST: th.vlseg2huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x23]

th.vlseg2huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x21]

th.vlseg2wuff.v v4, (a0)
# CHECK-INST: th.vlseg2wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x23]

th.vlseg2wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg2wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x23]

th.vlseg2wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x21]

th.vlseg2eff.v v4, (a0)
# CHECK-INST: th.vlseg2eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x23]

th.vlseg2eff.v v4, 0(a0)
# CHECK-INST: th.vlseg2eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x23]

th.vlseg2eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg2eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x21]

th.vlseg3bff.v v4, (a0)
# CHECK-INST: th.vlseg3bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x53]

th.vlseg3bff.v v4, 0(a0)
# CHECK-INST: th.vlseg3bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x53]

th.vlseg3bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x51]

th.vlseg3hff.v v4, (a0)
# CHECK-INST: th.vlseg3hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x53]

th.vlseg3hff.v v4, 0(a0)
# CHECK-INST: th.vlseg3hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x53]

th.vlseg3hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x51]

th.vlseg3wff.v v4, (a0)
# CHECK-INST: th.vlseg3wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x53]

th.vlseg3wff.v v4, 0(a0)
# CHECK-INST: th.vlseg3wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x53]

th.vlseg3wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x51]

th.vlseg3buff.v v4, (a0)
# CHECK-INST: th.vlseg3buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x43]

th.vlseg3buff.v v4, 0(a0)
# CHECK-INST: th.vlseg3buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x43]

th.vlseg3buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x41]

th.vlseg3huff.v v4, (a0)
# CHECK-INST: th.vlseg3huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x43]

th.vlseg3huff.v v4, 0(a0)
# CHECK-INST: th.vlseg3huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x43]

th.vlseg3huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x41]

th.vlseg3wuff.v v4, (a0)
# CHECK-INST: th.vlseg3wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x43]

th.vlseg3wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg3wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x43]

th.vlseg3wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x41]

th.vlseg3eff.v v4, (a0)
# CHECK-INST: th.vlseg3eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x43]

th.vlseg3eff.v v4, 0(a0)
# CHECK-INST: th.vlseg3eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x43]

th.vlseg3eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg3eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x41]

th.vlseg4bff.v v4, (a0)
# CHECK-INST: th.vlseg4bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x73]

th.vlseg4bff.v v4, 0(a0)
# CHECK-INST: th.vlseg4bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x73]

th.vlseg4bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x71]

th.vlseg4hff.v v4, (a0)
# CHECK-INST: th.vlseg4hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x73]

th.vlseg4hff.v v4, 0(a0)
# CHECK-INST: th.vlseg4hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x73]

th.vlseg4hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x71]

th.vlseg4wff.v v4, (a0)
# CHECK-INST: th.vlseg4wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x73]

th.vlseg4wff.v v4, 0(a0)
# CHECK-INST: th.vlseg4wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x73]

th.vlseg4wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x71]

th.vlseg4buff.v v4, (a0)
# CHECK-INST: th.vlseg4buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x63]

th.vlseg4buff.v v4, 0(a0)
# CHECK-INST: th.vlseg4buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x63]

th.vlseg4buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x61]

th.vlseg4huff.v v4, (a0)
# CHECK-INST: th.vlseg4huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x63]

th.vlseg4huff.v v4, 0(a0)
# CHECK-INST: th.vlseg4huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x63]

th.vlseg4huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x61]

th.vlseg4wuff.v v4, (a0)
# CHECK-INST: th.vlseg4wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x63]

th.vlseg4wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg4wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x63]

th.vlseg4wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x61]

th.vlseg4eff.v v4, (a0)
# CHECK-INST: th.vlseg4eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x63]

th.vlseg4eff.v v4, 0(a0)
# CHECK-INST: th.vlseg4eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x63]

th.vlseg4eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg4eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x61]

th.vlseg5bff.v v4, (a0)
# CHECK-INST: th.vlseg5bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x93]

th.vlseg5bff.v v4, 0(a0)
# CHECK-INST: th.vlseg5bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x93]

th.vlseg5bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x91]

th.vlseg5hff.v v4, (a0)
# CHECK-INST: th.vlseg5hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x93]

th.vlseg5hff.v v4, 0(a0)
# CHECK-INST: th.vlseg5hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x93]

th.vlseg5hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x91]

th.vlseg5wff.v v4, (a0)
# CHECK-INST: th.vlseg5wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x93]

th.vlseg5wff.v v4, 0(a0)
# CHECK-INST: th.vlseg5wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x93]

th.vlseg5wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x91]

th.vlseg5buff.v v4, (a0)
# CHECK-INST: th.vlseg5buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x83]

th.vlseg5buff.v v4, 0(a0)
# CHECK-INST: th.vlseg5buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0x83]

th.vlseg5buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0x81]

th.vlseg5huff.v v4, (a0)
# CHECK-INST: th.vlseg5huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x83]

th.vlseg5huff.v v4, 0(a0)
# CHECK-INST: th.vlseg5huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0x83]

th.vlseg5huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0x81]

th.vlseg5wuff.v v4, (a0)
# CHECK-INST: th.vlseg5wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x83]

th.vlseg5wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg5wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0x83]

th.vlseg5wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0x81]

th.vlseg5eff.v v4, (a0)
# CHECK-INST: th.vlseg5eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x83]

th.vlseg5eff.v v4, 0(a0)
# CHECK-INST: th.vlseg5eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0x83]

th.vlseg5eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg5eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0x81]

th.vlseg6bff.v v4, (a0)
# CHECK-INST: th.vlseg6bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xb3]

th.vlseg6bff.v v4, 0(a0)
# CHECK-INST: th.vlseg6bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xb3]

th.vlseg6bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xb1]

th.vlseg6hff.v v4, (a0)
# CHECK-INST: th.vlseg6hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xb3]

th.vlseg6hff.v v4, 0(a0)
# CHECK-INST: th.vlseg6hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xb3]

th.vlseg6hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xb1]

th.vlseg6wff.v v4, (a0)
# CHECK-INST: th.vlseg6wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xb3]

th.vlseg6wff.v v4, 0(a0)
# CHECK-INST: th.vlseg6wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xb3]

th.vlseg6wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xb1]

th.vlseg6buff.v v4, (a0)
# CHECK-INST: th.vlseg6buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xa3]

th.vlseg6buff.v v4, 0(a0)
# CHECK-INST: th.vlseg6buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xa3]

th.vlseg6buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xa1]

th.vlseg6huff.v v4, (a0)
# CHECK-INST: th.vlseg6huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xa3]

th.vlseg6huff.v v4, 0(a0)
# CHECK-INST: th.vlseg6huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xa3]

th.vlseg6huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xa1]

th.vlseg6wuff.v v4, (a0)
# CHECK-INST: th.vlseg6wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xa3]

th.vlseg6wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg6wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xa3]

th.vlseg6wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xa1]

th.vlseg6eff.v v4, (a0)
# CHECK-INST: th.vlseg6eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xa3]

th.vlseg6eff.v v4, 0(a0)
# CHECK-INST: th.vlseg6eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xa3]

th.vlseg6eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg6eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xa1]

th.vlseg7bff.v v4, (a0)
# CHECK-INST: th.vlseg7bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xd3]

th.vlseg7bff.v v4, 0(a0)
# CHECK-INST: th.vlseg7bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xd3]

th.vlseg7bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xd1]

th.vlseg7hff.v v4, (a0)
# CHECK-INST: th.vlseg7hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xd3]

th.vlseg7hff.v v4, 0(a0)
# CHECK-INST: th.vlseg7hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xd3]

th.vlseg7hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xd1]

th.vlseg7wff.v v4, (a0)
# CHECK-INST: th.vlseg7wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xd3]

th.vlseg7wff.v v4, 0(a0)
# CHECK-INST: th.vlseg7wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xd3]

th.vlseg7wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xd1]

th.vlseg7buff.v v4, (a0)
# CHECK-INST: th.vlseg7buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xc3]

th.vlseg7buff.v v4, 0(a0)
# CHECK-INST: th.vlseg7buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xc3]

th.vlseg7buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xc1]

th.vlseg7huff.v v4, (a0)
# CHECK-INST: th.vlseg7huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xc3]

th.vlseg7huff.v v4, 0(a0)
# CHECK-INST: th.vlseg7huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xc3]

th.vlseg7huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xc1]

th.vlseg7wuff.v v4, (a0)
# CHECK-INST: th.vlseg7wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xc3]

th.vlseg7wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg7wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xc3]

th.vlseg7wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xc1]

th.vlseg7eff.v v4, (a0)
# CHECK-INST: th.vlseg7eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xc3]

th.vlseg7eff.v v4, 0(a0)
# CHECK-INST: th.vlseg7eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xc3]

th.vlseg7eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg7eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xc1]

th.vlseg8bff.v v4, (a0)
# CHECK-INST: th.vlseg8bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xf3]

th.vlseg8bff.v v4, 0(a0)
# CHECK-INST: th.vlseg8bff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xf3]

th.vlseg8bff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8bff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xf1]

th.vlseg8hff.v v4, (a0)
# CHECK-INST: th.vlseg8hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xf3]

th.vlseg8hff.v v4, 0(a0)
# CHECK-INST: th.vlseg8hff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xf3]

th.vlseg8hff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8hff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xf1]

th.vlseg8wff.v v4, (a0)
# CHECK-INST: th.vlseg8wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xf3]

th.vlseg8wff.v v4, 0(a0)
# CHECK-INST: th.vlseg8wff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xf3]

th.vlseg8wff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8wff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xf1]

th.vlseg8buff.v v4, (a0)
# CHECK-INST: th.vlseg8buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xe3]

th.vlseg8buff.v v4, 0(a0)
# CHECK-INST: th.vlseg8buff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x02,0x05,0xe3]

th.vlseg8buff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8buff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x02,0x05,0xe1]

th.vlseg8huff.v v4, (a0)
# CHECK-INST: th.vlseg8huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xe3]

th.vlseg8huff.v v4, 0(a0)
# CHECK-INST: th.vlseg8huff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x52,0x05,0xe3]

th.vlseg8huff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8huff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x52,0x05,0xe1]

th.vlseg8wuff.v v4, (a0)
# CHECK-INST: th.vlseg8wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xe3]

th.vlseg8wuff.v v4, 0(a0)
# CHECK-INST: th.vlseg8wuff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x62,0x05,0xe3]

th.vlseg8wuff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8wuff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x62,0x05,0xe1]

th.vlseg8eff.v v4, (a0)
# CHECK-INST: th.vlseg8eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xe3]

th.vlseg8eff.v v4, 0(a0)
# CHECK-INST: th.vlseg8eff.v	v4, (a0)
# CHECK-ENCODING: [0x07,0x72,0x05,0xe3]

th.vlseg8eff.v v4, (a0), v0.t
# CHECK-INST: th.vlseg8eff.v	v4, (a0), v0.t
# CHECK-ENCODING: [0x07,0x72,0x05,0xe1]

th.vamoaddw.v v4, v8, (a1), v4
# CHECK-INST: th.vamoaddw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x06]

th.vamoaddw.v zero, v8, (a1), v4
# CHECK-INST: th.vamoaddw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x02]

th.vamoaddd.v v4, v8, (a1), v4
# CHECK-INST: th.vamoaddd.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x06]

th.vamoaddd.v zero, v8, (a1), v4
# CHECK-INST: th.vamoaddd.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x02]

th.vamoaddw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoaddw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x04]

th.vamoaddw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoaddw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x00]

th.vamoaddd.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoaddd.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x04]

th.vamoaddd.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoaddd.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x00]

th.vamoswapw.v v4, v8, (a1), v4
# CHECK-INST: th.vamoswapw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x0e]

th.vamoswapw.v zero, v8, (a1), v4
# CHECK-INST: th.vamoswapw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x0a]

th.vamoswapd.v v4, v8, (a1), v4
# CHECK-INST: th.vamoswapd.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x0e]

th.vamoswapd.v zero, v8, (a1), v4
# CHECK-INST: th.vamoswapd.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x0a]

th.vamoswapw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoswapw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x0c]

th.vamoswapw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoswapw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x08]

th.vamoswapd.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoswapd.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x0c]

th.vamoswapd.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoswapd.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x08]

th.vamoxorw.v v4, v8, (a1), v4
# CHECK-INST: th.vamoxorw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x26]

th.vamoxorw.v zero, v8, (a1), v4
# CHECK-INST: th.vamoxorw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x22]

th.vamoxord.v v4, v8, (a1), v4
# CHECK-INST: th.vamoxord.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x26]

th.vamoxord.v zero, v8, (a1), v4
# CHECK-INST: th.vamoxord.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x22]

th.vamoxorw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoxorw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x24]

th.vamoxorw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoxorw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x20]

th.vamoxord.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoxord.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x24]

th.vamoxord.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoxord.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x20]

th.vamoandw.v v4, v8, (a1), v4
# CHECK-INST: th.vamoandw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x66]

th.vamoandw.v zero, v8, (a1), v4
# CHECK-INST: th.vamoandw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x62]

th.vamoandd.v v4, v8, (a1), v4
# CHECK-INST: th.vamoandd.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x66]

th.vamoandd.v zero, v8, (a1), v4
# CHECK-INST: th.vamoandd.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x62]

th.vamoandw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoandw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x64]

th.vamoandw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoandw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x60]

th.vamoandd.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoandd.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x64]

th.vamoandd.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoandd.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x60]

th.vamoorw.v v4, v8, (a1), v4
# CHECK-INST: th.vamoorw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x46]

th.vamoorw.v zero, v8, (a1), v4
# CHECK-INST: th.vamoorw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x42]

th.vamoord.v v4, v8, (a1), v4
# CHECK-INST: th.vamoord.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x46]

th.vamoord.v zero, v8, (a1), v4
# CHECK-INST: th.vamoord.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x42]

th.vamoorw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoorw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x44]

th.vamoorw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoorw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x40]

th.vamoord.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoord.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x44]

th.vamoord.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamoord.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x40]

th.vamominw.v v4, v8, (a1), v4
# CHECK-INST: th.vamominw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x86]

th.vamominw.v zero, v8, (a1), v4
# CHECK-INST: th.vamominw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x82]

th.vamomind.v v4, v8, (a1), v4
# CHECK-INST: th.vamomind.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x86]

th.vamomind.v zero, v8, (a1), v4
# CHECK-INST: th.vamomind.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x82]

th.vamominw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x84]

th.vamominw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0x80]

th.vamomind.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomind.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x84]

th.vamomind.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomind.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0x80]

th.vamomaxw.v v4, v8, (a1), v4
# CHECK-INST: th.vamomaxw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xa6]

th.vamomaxw.v zero, v8, (a1), v4
# CHECK-INST: th.vamomaxw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xa2]

th.vamomaxd.v v4, v8, (a1), v4
# CHECK-INST: th.vamomaxd.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xa6]

th.vamomaxd.v zero, v8, (a1), v4
# CHECK-INST: th.vamomaxd.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xa2]

th.vamomaxw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xa4]

th.vamomaxw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xa0]

th.vamomaxd.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxd.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xa4]

th.vamomaxd.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxd.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xa0]

th.vamominuw.v v4, v8, (a1), v4
# CHECK-INST: th.vamominuw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xc6]

th.vamominuw.v zero, v8, (a1), v4
# CHECK-INST: th.vamominuw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xc2]

th.vamominud.v v4, v8, (a1), v4
# CHECK-INST: th.vamominud.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xc6]

th.vamominud.v zero, v8, (a1), v4
# CHECK-INST: th.vamominud.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xc2]

th.vamominuw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominuw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xc4]

th.vamominuw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominuw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xc0]

th.vamominud.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominud.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xc4]

th.vamominud.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamominud.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xc0]

th.vamomaxuw.v v4, v8, (a1), v4
# CHECK-INST: th.vamomaxuw.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xe6]

th.vamomaxuw.v zero, v8, (a1), v4
# CHECK-INST: th.vamomaxuw.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xe2]

th.vamomaxud.v v4, v8, (a1), v4
# CHECK-INST: th.vamomaxud.v v4, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xe6]

th.vamomaxud.v zero, v8, (a1), v4
# CHECK-INST: th.vamomaxud.v x0, v8, (a1), v4
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xe2]

th.vamomaxuw.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxuw.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xe4]

th.vamomaxuw.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxuw.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xe2,0x85,0xe0]

th.vamomaxud.v v4, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxud.v v4, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xe4]

th.vamomaxud.v zero, v8, (a1), v4, v0.t
# CHECK-INST: th.vamomaxud.v x0, v8, (a1), v4, v0.t
# CHECK-ENCODING: [0x2f,0xf2,0x85,0xe0]

th.vadd.vv v4, v8, v12
# CHECK-INST: th.vadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x02]

th.vadd.vx v4, v8, a1
# CHECK-INST: th.vadd.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x02]

th.vadd.vi v4, v8, 15
# CHECK-INST: th.vadd.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x02]

th.vadd.vi v4, v8, -16
# CHECK-INST: th.vadd.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x02]

th.vadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x00]

th.vadd.vx v4, v8, a1, v0.t
# CHECK-INST: th.vadd.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x00]

th.vadd.vi v4, v8, 15, v0.t
# CHECK-INST: th.vadd.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x00]

th.vadd.vi v4, v8, -16, v0.t
# CHECK-INST: th.vadd.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x00]

th.vsub.vv v4, v8, v12
# CHECK-INST: th.vsub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x0a]

th.vsub.vx v4, v8, a1
# CHECK-INST: th.vsub.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x0a]

th.vrsub.vx v4, v8, a1
# CHECK-INST: th.vrsub.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x0e]

th.vrsub.vi v4, v8, 15
# CHECK-INST: th.vrsub.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x0e]

th.vrsub.vi v4, v8, -16
# CHECK-INST: th.vrsub.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x0e]

th.vsub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x08]

th.vsub.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsub.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x08]

th.vrsub.vx v4, v8, a1, v0.t
# CHECK-INST: th.vrsub.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x0c]

th.vrsub.vi v4, v8, 15, v0.t
# CHECK-INST: th.vrsub.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x0c]

th.vrsub.vi v4, v8, -16, v0.t
# CHECK-INST: th.vrsub.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x0c]

th.vwcvt.x.x.v v4, v8
# CHECK-INST: th.vwcvt.x.x.v v4, v8
# CHECK-ENCODING: [0x57,0x62,0x80,0xc6]

th.vwcvtu.x.x.v v4, v8
# CHECK-INST: th.vwcvtu.x.x.v v4, v8
# CHECK-ENCODING: [0x57,0x62,0x80,0xc2]

th.vwcvt.x.x.v v4, v8, v0.t
# CHECK-INST: th.vwcvt.x.x.v v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x62,0x80,0xc4]

th.vwcvtu.x.x.v v4, v8, v0.t
# CHECK-INST: th.vwcvtu.x.x.v v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x62,0x80,0xc0]

th.vwaddu.vv v4, v8, v12
# CHECK-INST: th.vwaddu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xc2]

th.vwaddu.vx v4, v8, a1
# CHECK-INST: th.vwaddu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xc2]

th.vwaddu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwaddu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xc0]

th.vwaddu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwaddu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xc0]

th.vwsubu.vv v4, v8, v12
# CHECK-INST: th.vwsubu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xca]

th.vwsubu.vx v4, v8, a1
# CHECK-INST: th.vwsubu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xca]

th.vwsubu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwsubu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xc8]

th.vwsubu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwsubu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xc8]

th.vwadd.vv v4, v8, v12
# CHECK-INST: th.vwadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xc6]

th.vwadd.vx v4, v8, a1
# CHECK-INST: th.vwadd.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xc6]

th.vwadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xc4]

th.vwadd.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwadd.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xc4]

th.vwsub.vv v4, v8, v12
# CHECK-INST: th.vwsub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xce]

th.vwsub.vx v4, v8, a1
# CHECK-INST: th.vwsub.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xce]

th.vwsub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwsub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xcc]

th.vwsub.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwsub.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xcc]

th.vwaddu.wv v4, v8, v12
# CHECK-INST: th.vwaddu.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xd2]

th.vwaddu.wx v4, v8, a1
# CHECK-INST: th.vwaddu.wx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xd2]

th.vwaddu.wv v4, v8, v12, v0.t
# CHECK-INST: th.vwaddu.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xd0]

th.vwaddu.wx v4, v8, a1, v0.t
# CHECK-INST: th.vwaddu.wx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xd0]

th.vwsubu.wv v4, v8, v12
# CHECK-INST: th.vwsubu.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xda]

th.vwsubu.wx v4, v8, a1
# CHECK-INST: th.vwsubu.wx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xda]

th.vwsubu.wv v4, v8, v12, v0.t
# CHECK-INST: th.vwsubu.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xd8]

th.vwsubu.wx v4, v8, a1, v0.t
# CHECK-INST: th.vwsubu.wx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xd8]

th.vwadd.wv v4, v8, v12
# CHECK-INST: th.vwadd.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xd6]

th.vwadd.wx v4, v8, a1
# CHECK-INST: th.vwadd.wx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xd6]

th.vwadd.wv v4, v8, v12, v0.t
# CHECK-INST: th.vwadd.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xd4]

th.vwadd.wx v4, v8, a1, v0.t
# CHECK-INST: th.vwadd.wx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xd4]

th.vwsub.wv v4, v8, v12
# CHECK-INST: th.vwsub.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xde]

th.vwsub.wx v4, v8, a1
# CHECK-INST: th.vwsub.wx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xde]

th.vwsub.wv v4, v8, v12, v0.t
# CHECK-INST: th.vwsub.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xdc]

th.vwsub.wx v4, v8, a1, v0.t
# CHECK-INST: th.vwsub.wx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xdc]

th.vadc.vvm v4, v8, v12, v0
# CHECK-INST: th.vadc.vvm	v4, v8, v12, v0
# CHECK-ENCODING: [0x57,0x02,0x86,0x42]

th.vadc.vxm v4, v8, a1, v0
# CHECK-INST: th.vadc.vxm	v4, v8, a1, v0
# CHECK-ENCODING: [0x57,0xc2,0x85,0x42]

th.vadc.vim v4, v8, 15, v0
# CHECK-INST: th.vadc.vim	v4, v8, 15, v0
# CHECK-ENCODING: [0x57,0xb2,0x87,0x42]

th.vadc.vim v4, v8, -16, v0
# CHECK-INST: th.vadc.vim	v4, v8, -16, v0
# CHECK-ENCODING: [0x57,0x32,0x88,0x42]

th.vmadc.vvm v4, v8, v12, v0
# CHECK-INST: th.vmadc.vvm	v4, v8, v12, v0
# CHECK-ENCODING: [0x57,0x02,0x86,0x46]

th.vmadc.vxm v4, v8, a1, v0
# CHECK-INST: th.vmadc.vxm	v4, v8, a1, v0
# CHECK-ENCODING: [0x57,0xc2,0x85,0x46]

th.vmadc.vim v4, v8, 15, v0
# CHECK-INST: th.vmadc.vim	v4, v8, 15, v0
# CHECK-ENCODING: [0x57,0xb2,0x87,0x46]

th.vmadc.vim v4, v8, -16, v0
# CHECK-INST: th.vmadc.vim	v4, v8, -16, v0
# CHECK-ENCODING: [0x57,0x32,0x88,0x46]

th.vsbc.vvm v4, v8, v12, v0
# CHECK-INST: th.vsbc.vvm	v4, v8, v12, v0
# CHECK-ENCODING: [0x57,0x02,0x86,0x4a]

th.vsbc.vxm v4, v8, a1, v0
# CHECK-INST: th.vsbc.vxm	v4, v8, a1, v0
# CHECK-ENCODING: [0x57,0xc2,0x85,0x4a]

th.vmsbc.vvm v4, v8, v12, v0
# CHECK-INST: th.vmsbc.vvm	v4, v8, v12, v0
# CHECK-ENCODING: [0x57,0x02,0x86,0x4e]

th.vmsbc.vxm v4, v8, a1, v0
# CHECK-INST: th.vmsbc.vxm	v4, v8, a1, v0
# CHECK-ENCODING: [0x57,0xc2,0x85,0x4e]

th.vnot.v v4, v8
# CHECK-INST: th.vnot.v v4, v8
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x2e]

th.vnot.v v4, v8, v0.t
# CHECK-INST: th.vnot.v v4, v8, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x2c]

th.vand.vv v4, v8, v12
# CHECK-INST: th.vand.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x26]

th.vand.vx v4, v8, a1
# CHECK-INST: th.vand.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x26]

th.vand.vi v4, v8, 15
# CHECK-INST: th.vand.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x26]

th.vand.vi v4, v8, -16
# CHECK-INST: th.vand.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x26]

th.vand.vv v4, v8, v12, v0.t
# CHECK-INST: th.vand.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x24]

th.vand.vx v4, v8, a1, v0.t
# CHECK-INST: th.vand.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x24]

th.vand.vi v4, v8, 15, v0.t
# CHECK-INST: th.vand.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x24]

th.vand.vi v4, v8, -16, v0.t
# CHECK-INST: th.vand.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x24]

th.vor.vv v4, v8, v12
# CHECK-INST: th.vor.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x2a]

th.vor.vx v4, v8, a1
# CHECK-INST: th.vor.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x2a]

th.vor.vi v4, v8, 15
# CHECK-INST: th.vor.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x2a]

th.vor.vi v4, v8, -16
# CHECK-INST: th.vor.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x2a]

th.vor.vv v4, v8, v12, v0.t
# CHECK-INST: th.vor.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x28]

th.vor.vx v4, v8, a1, v0.t
# CHECK-INST: th.vor.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x28]

th.vor.vi v4, v8, 15, v0.t
# CHECK-INST: th.vor.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x28]

th.vor.vi v4, v8, -16, v0.t
# CHECK-INST: th.vor.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x28]

th.vxor.vv v4, v8, v12
# CHECK-INST: th.vxor.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x2e]

th.vxor.vx v4, v8, a1
# CHECK-INST: th.vxor.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x2e]

th.vxor.vi v4, v8, 15
# CHECK-INST: th.vxor.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x2e]

th.vxor.vi v4, v8, -16
# CHECK-INST: th.vxor.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x2e]

th.vxor.vv v4, v8, v12, v0.t
# CHECK-INST: th.vxor.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x2c]

th.vxor.vx v4, v8, a1, v0.t
# CHECK-INST: th.vxor.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x2c]

th.vxor.vi v4, v8, 15, v0.t
# CHECK-INST: th.vxor.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x2c]

th.vxor.vi v4, v8, -16, v0.t
# CHECK-INST: th.vxor.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x2c]

th.vsll.vv v4, v8, v12
# CHECK-INST: th.vsll.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x96]

th.vsll.vx v4, v8, a1
# CHECK-INST: th.vsll.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x96]

th.vsll.vi v4, v8, 1
# CHECK-INST: th.vsll.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0x96]

th.vsll.vi v4, v8, 31
# CHECK-INST: th.vsll.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x96]

th.vsll.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsll.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x94]

th.vsll.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsll.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x94]

th.vsll.vi v4, v8, 1, v0.t
# CHECK-INST: th.vsll.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0x94]

th.vsll.vi v4, v8, 31, v0.t
# CHECK-INST: th.vsll.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x94]

th.vsrl.vv v4, v8, v12
# CHECK-INST: th.vsrl.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xa2]

th.vsrl.vx v4, v8, a1
# CHECK-INST: th.vsrl.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xa2]

th.vsrl.vi v4, v8, 1
# CHECK-INST: th.vsrl.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xa2]

th.vsrl.vi v4, v8, 31
# CHECK-INST: th.vsrl.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xa2]

th.vsrl.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsrl.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xa0]

th.vsrl.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsrl.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xa0]

th.vsrl.vi v4, v8, 1, v0.t
# CHECK-INST: th.vsrl.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xa0]

th.vsrl.vi v4, v8, 31, v0.t
# CHECK-INST: th.vsrl.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xa0]

th.vsra.vv v4, v8, v12
# CHECK-INST: th.vsra.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xa6]

th.vsra.vx v4, v8, a1
# CHECK-INST: th.vsra.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xa6]

th.vsra.vi v4, v8, 1
# CHECK-INST: th.vsra.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xa6]

th.vsra.vi v4, v8, 31
# CHECK-INST: th.vsra.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xa6]

th.vsra.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsra.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xa4]

th.vsra.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsra.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xa4]

th.vsra.vi v4, v8, 1, v0.t
# CHECK-INST: th.vsra.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xa4]

th.vsra.vi v4, v8, 31, v0.t
# CHECK-INST: th.vsra.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xa4]

th.vnsrl.vv v4, v8, v12
# CHECK-INST: th.vnsrl.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xb2]

th.vnsrl.vx v4, v8, a1
# CHECK-INST: th.vnsrl.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xb2]

th.vnsrl.vi v4, v8, 1
# CHECK-INST: th.vnsrl.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xb2]

th.vnsrl.vi v4, v8, 31
# CHECK-INST: th.vnsrl.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xb2]

th.vnsrl.vv v4, v8, v12, v0.t
# CHECK-INST: th.vnsrl.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xb0]

th.vnsrl.vx v4, v8, a1, v0.t
# CHECK-INST: th.vnsrl.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xb0]

th.vnsrl.vi v4, v8, 1, v0.t
# CHECK-INST: th.vnsrl.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xb0]

th.vnsrl.vi v4, v8, 31, v0.t
# CHECK-INST: th.vnsrl.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xb0]

th.vnsra.vv v4, v8, v12
# CHECK-INST: th.vnsra.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xb6]

th.vnsra.vx v4, v8, a1
# CHECK-INST: th.vnsra.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xb6]

th.vnsra.vi v4, v8, 1
# CHECK-INST: th.vnsra.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xb6]

th.vnsra.vi v4, v8, 31
# CHECK-INST: th.vnsra.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xb6]

th.vnsra.vv v4, v8, v12, v0.t
# CHECK-INST: th.vnsra.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xb4]

th.vnsra.vx v4, v8, a1, v0.t
# CHECK-INST: th.vnsra.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xb4]

th.vnsra.vi v4, v8, 1, v0.t
# CHECK-INST: th.vnsra.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xb4]

th.vnsra.vi v4, v8, 31, v0.t
# CHECK-INST: th.vnsra.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xb4]

th.vmsgt.vv v4, v8, v12
# CHECK-INST: th.vmslt.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0xc4,0x6e]

th.vmsgtu.vv v4, v8, v12
# CHECK-INST: th.vmsltu.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0xc4,0x6a]

th.vmsge.vv v4, v8, v12
# CHECK-INST: th.vmsle.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0xc4,0x76]

th.vmsgeu.vv v4, v8, v12
# CHECK-INST: th.vmsleu.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0xc4,0x72]

th.vmsgt.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmslt.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0xc4,0x6c]

th.vmsgtu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsltu.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0xc4,0x68]

th.vmsge.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsle.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0xc4,0x74]

th.vmsgeu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsleu.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0xc4,0x70]

th.vmsge.vx v4, v8, a1
# CHECK-INST: th.vmslt.vx v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x6e]
# CHECK-INST: th.vmnot.m v4, v4
# CHECK-ENCODING: [0x57,0x22,0x42,0x76]

th.vmsgeu.vx v4, v8, a1
# CHECK-INST: th.vmsltu.vx v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x6a]
# CHECK-INST: th.vmnot.m v4, v4
# CHECK-ENCODING: [0x57,0x22,0x42,0x76]

th.vmsge.vx v8, v12, a2, v0.t
# CHECK-INST: th.vmslt.vx v8, v12, a2, v0.t
# CHECK-ENCODING: [0x57,0x44,0xc6,0x6c]
# CHECK-INST: th.vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]

th.vmsgeu.vx v8, v12, a2, v0.t
# CHECK-INST: th.vmsltu.vx v8, v12, a2, v0.t
# CHECK-ENCODING: [0x57,0x44,0xc6,0x68]
# CHECK-INST: th.vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]

th.vmsge.vx v4, v8, a1, v0.t, v12
# CHECK-INST: th.vmslt.vx v12, v8, a1
# TODO: GCC produces vmslt.vx v12, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc6,0x85,0x6e]
# CHECK-INST: th.vmandnot.mm v4, v4, v12
# CHECK-ENCODING: [0x57,0x22,0x46,0x62]

th.vmsgeu.vx v4, v8, a1, v0.t, v12
# CHECK-INST: th.vmsltu.vx v12, v8, a1
# TODO: GCC produces vmsltu.vx v12, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc6,0x85,0x6a]
# CHECK-INST: th.vmandnot.mm v4, v4, v12
# CHECK-ENCODING: [0x57,0x22,0x46,0x62]

th.vmslt.vi v4, v8, 16
# CHECK-INST: th.vmsle.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x76]

th.vmslt.vi v4, v8, -15
# CHECK-INST: th.vmsle.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x76]

th.vmsltu.vi v4, v8, 16
# CHECK-INST: th.vmsleu.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x72]

th.vmsltu.vi v4, v8, -15
# CHECK-INST: th.vmsleu.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x72]

th.vmsge.vi v4, v8, 16
# CHECK-INST: th.vmsgt.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7e]

th.vmsge.vi v4, v8, -15
# CHECK-INST: th.vmsgt.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x7e]

th.vmsgeu.vi v4, v8, 16
# CHECK-INST: th.vmsgtu.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7a]

th.vmsgeu.vi v4, v8, -15
# CHECK-INST: th.vmsgtu.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x7a]

th.vmslt.vi v4, v8, 16, v0.t
# CHECK-INST: th.vmsle.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x74]

th.vmslt.vi v4, v8, -15, v0.t
# CHECK-INST: th.vmsle.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x74]

th.vmsltu.vi v4, v8, 16, v0.t
# CHECK-INST: th.vmsleu.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x70]

th.vmsltu.vi v4, v8, -15, v0.t
# CHECK-INST: th.vmsleu.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x70]

th.vmsge.vi v4, v8, 16, v0.t
# CHECK-INST: th.vmsgt.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7c]

th.vmsge.vi v4, v8, -15, v0.t
# CHECK-INST: th.vmsgt.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x7c]

th.vmsgeu.vi v4, v8, 16, v0.t
# CHECK-INST: th.vmsgtu.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x78]

th.vmsgeu.vi v4, v8, -15, v0.t
# CHECK-INST: th.vmsgtu.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x78]

th.vmseq.vv v4, v8, v12
# CHECK-INST: th.vmseq.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x62]

th.vmseq.vx v4, v8, a1
# CHECK-INST: th.vmseq.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x62]

th.vmseq.vi v4, v8, 15
# CHECK-INST: th.vmseq.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x62]

th.vmseq.vi v4, v8, -16
# CHECK-INST: th.vmseq.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x62]

th.vmseq.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmseq.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x60]

th.vmseq.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmseq.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x60]

th.vmseq.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmseq.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x60]

th.vmseq.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmseq.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x60]

th.vmsne.vv v4, v8, v12
# CHECK-INST: th.vmsne.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x66]

th.vmsne.vx v4, v8, a1
# CHECK-INST: th.vmsne.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x66]

th.vmsne.vi v4, v8, 15
# CHECK-INST: th.vmsne.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x66]

th.vmsne.vi v4, v8, -16
# CHECK-INST: th.vmsne.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x66]

th.vmsne.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsne.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x64]

th.vmsne.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsne.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x64]

th.vmsne.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmsne.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x64]

th.vmsne.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmsne.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x64]

th.vmsltu.vv v4, v8, v12
# CHECK-INST: th.vmsltu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x6a]

th.vmsltu.vx v4, v8, a1
# CHECK-INST: th.vmsltu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x6a]

th.vmsltu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsltu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x68]

th.vmsltu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsltu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x68]

th.vmslt.vv v4, v8, v12
# CHECK-INST: th.vmslt.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x6e]

th.vmslt.vx v4, v8, a1
# CHECK-INST: th.vmslt.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x6e]

th.vmslt.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmslt.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x6c]

th.vmslt.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmslt.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x6c]

th.vmsleu.vv v4, v8, v12
# CHECK-INST: th.vmsleu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x72]

th.vmsleu.vx v4, v8, a1
# CHECK-INST: th.vmsleu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x72]

th.vmsleu.vi v4, v8, 15
# CHECK-INST: th.vmsleu.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x72]

th.vmsleu.vi v4, v8, -16
# CHECK-INST: th.vmsleu.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x72]

th.vmsleu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsleu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x70]

th.vmsleu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsleu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x70]

th.vmsleu.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmsleu.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x70]

th.vmsleu.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmsleu.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x70]

th.vmsle.vv v4, v8, v12
# CHECK-INST: th.vmsle.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x76]

th.vmsle.vx v4, v8, a1
# CHECK-INST: th.vmsle.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x76]

th.vmsle.vi v4, v8, 15
# CHECK-INST: th.vmsle.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x76]

th.vmsle.vi v4, v8, -16
# CHECK-INST: th.vmsle.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x76]

th.vmsle.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmsle.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x74]

th.vmsle.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsle.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x74]

th.vmsle.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmsle.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x74]

th.vmsle.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmsle.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x74]

th.vmsgtu.vx v4, v8, a1
# CHECK-INST: th.vmsgtu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x7a]

th.vmsgtu.vi v4, v8, 15
# CHECK-INST: th.vmsgtu.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7a]

th.vmsgtu.vi v4, v8, -16
# CHECK-INST: th.vmsgtu.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x7a]

th.vmsgtu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsgtu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x78]

th.vmsgtu.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmsgtu.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x78]

th.vmsgtu.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmsgtu.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x78]

th.vmsgt.vx v4, v8, a1
# CHECK-INST: th.vmsgt.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x7e]

th.vmsgt.vi v4, v8, 15
# CHECK-INST: th.vmsgt.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7e]

th.vmsgt.vi v4, v8, -16
# CHECK-INST: th.vmsgt.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x7e]

th.vmsgt.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmsgt.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x7c]

th.vmsgt.vi v4, v8, 15, v0.t
# CHECK-INST: th.vmsgt.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x7c]

th.vmsgt.vi v4, v8, -16, v0.t
# CHECK-INST: th.vmsgt.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x7c]

th.vminu.vv v4, v8, v12
# CHECK-INST: th.vminu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x12]

th.vminu.vx v4, v8, a1
# CHECK-INST: th.vminu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x12]

th.vminu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vminu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x10]

th.vminu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vminu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x10]

th.vmin.vv v4, v8, v12
# CHECK-INST: th.vmin.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x16]

th.vmin.vx v4, v8, a1
# CHECK-INST: th.vmin.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x16]

th.vmin.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmin.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x14]

th.vmin.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmin.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x14]

th.vmaxu.vv v4, v8, v12
# CHECK-INST: th.vmaxu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x1a]

th.vmaxu.vx v4, v8, a1
# CHECK-INST: th.vmaxu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x1a]

th.vmaxu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmaxu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x18]

th.vmaxu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmaxu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x18]

th.vmax.vv v4, v8, v12
# CHECK-INST: th.vmax.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x1e]

th.vmax.vx v4, v8, a1
# CHECK-INST: th.vmax.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x1e]

th.vmax.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmax.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x1c]

th.vmax.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmax.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x1c]

th.vmul.vv v4, v8, v12
# CHECK-INST: th.vmul.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x96]

th.vmul.vx v4, v8, a1
# CHECK-INST: th.vmul.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x96]

th.vmul.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmul.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x94]

th.vmul.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmul.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x94]

th.vmulh.vv v4, v8, v12
# CHECK-INST: th.vmulh.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x9e]

th.vmulh.vx v4, v8, a1
# CHECK-INST: th.vmulh.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x9e]

th.vmulh.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmulh.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x9c]

th.vmulh.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmulh.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x9c]

th.vmulhu.vv v4, v8, v12
# CHECK-INST: th.vmulhu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x92]

th.vmulhu.vx v4, v8, a1
# CHECK-INST: th.vmulhu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x92]

th.vmulhu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmulhu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x90]

th.vmulhu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmulhu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x90]

th.vmulhsu.vv v4, v8, v12
# CHECK-INST: th.vmulhsu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x9a]

th.vmulhsu.vx v4, v8, a1
# CHECK-INST: th.vmulhsu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x9a]

th.vmulhsu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmulhsu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x98]

th.vmulhsu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vmulhsu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x98]

th.vwmul.vv v4, v8, v12
# CHECK-INST: th.vwmul.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xee]

th.vwmul.vx v4, v8, a1
# CHECK-INST: th.vwmul.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xee]

th.vwmul.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwmul.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xec]

th.vwmul.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwmul.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xec]

th.vwmulu.vv v4, v8, v12
# CHECK-INST: th.vwmulu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xe2]

th.vwmulu.vx v4, v8, a1
# CHECK-INST: th.vwmulu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xe2]

th.vwmulu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwmulu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xe0]

th.vwmulu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwmulu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xe0]

th.vwmulsu.vv v4, v8, v12
# CHECK-INST: th.vwmulsu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0xea]

th.vwmulsu.vx v4, v8, a1
# CHECK-INST: th.vwmulsu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0xea]

th.vwmulsu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vwmulsu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xe8]

th.vwmulsu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vwmulsu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xe8]

th.vmacc.vv v4, v12, v8
# CHECK-INST: th.vmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xb6]

th.vmacc.vx v4, a1, v8
# CHECK-INST: th.vmacc.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xb6]

th.vmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xb4]

th.vmacc.vx v4, a1, v8, v0.t
# CHECK-INST: th.vmacc.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xb4]

th.vnmsac.vv v4, v12, v8
# CHECK-INST: th.vnmsac.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xbe]

th.vnmsac.vx v4, a1, v8
# CHECK-INST: th.vnmsac.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xbe]

th.vnmsac.vv v4, v12, v8, v0.t
# CHECK-INST: th.vnmsac.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xbc]

th.vnmsac.vx v4, a1, v8, v0.t
# CHECK-INST: th.vnmsac.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xbc]

th.vmadd.vv v4, v12, v8
# CHECK-INST: th.vmadd.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xa6]

th.vmadd.vx v4, a1, v8
# CHECK-INST: th.vmadd.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xa6]

th.vmadd.vv v4, v12, v8, v0.t
# CHECK-INST: th.vmadd.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xa4]

th.vmadd.vx v4, a1, v8, v0.t
# CHECK-INST: th.vmadd.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xa4]

th.vnmsub.vv v4, v12, v8
# CHECK-INST: th.vnmsub.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xae]

th.vnmsub.vx v4, a1, v8
# CHECK-INST: th.vnmsub.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xae]

th.vnmsub.vv v4, v12, v8, v0.t
# CHECK-INST: th.vnmsub.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xac]

th.vnmsub.vx v4, a1, v8, v0.t
# CHECK-INST: th.vnmsub.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xac]

th.vwmaccu.vv v4, v12, v8
# CHECK-INST: th.vwmaccu.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xf2]

th.vwmaccu.vx v4, a1, v8
# CHECK-INST: th.vwmaccu.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xf2]

th.vwmaccu.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwmaccu.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xf0]

th.vwmaccu.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwmaccu.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xf0]

th.vwmacc.vv v4, v12, v8
# CHECK-INST: th.vwmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xf6]

th.vwmacc.vx v4, a1, v8
# CHECK-INST: th.vwmacc.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xf6]

th.vwmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xf4]

th.vwmacc.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwmacc.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xf4]

th.vwmaccsu.vv v4, v12, v8
# CHECK-INST: th.vwmaccsu.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x22,0x86,0xfa]

th.vwmaccsu.vx v4, a1, v8
# CHECK-INST: th.vwmaccsu.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xfa]

th.vwmaccsu.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwmaccsu.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0xf8]

th.vwmaccsu.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwmaccsu.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xf8]

th.vwmaccus.vx v4, a1, v8
# CHECK-INST: th.vwmaccus.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xe2,0x85,0xfe]

th.vwmaccus.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwmaccus.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0xfc]

th.vdivu.vv v4, v8, v12
# CHECK-INST: th.vdivu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x82]

th.vdivu.vx v4, v8, a1
# CHECK-INST: th.vdivu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x82]

th.vdivu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vdivu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x80]

th.vdivu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vdivu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x80]

th.vdiv.vv v4, v8, v12
# CHECK-INST: th.vdiv.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x86]

th.vdiv.vx v4, v8, a1
# CHECK-INST: th.vdiv.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x86]

th.vdiv.vv v4, v8, v12, v0.t
# CHECK-INST: th.vdiv.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x84]

th.vdiv.vx v4, v8, a1, v0.t
# CHECK-INST: th.vdiv.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x84]

th.vremu.vv v4, v8, v12
# CHECK-INST: th.vremu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x8a]

th.vremu.vx v4, v8, a1
# CHECK-INST: th.vremu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x8a]

th.vremu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vremu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x88]

th.vremu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vremu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x88]

th.vrem.vv v4, v8, v12
# CHECK-INST: th.vrem.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x8e]

th.vrem.vx v4, v8, a1
# CHECK-INST: th.vrem.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x8e]

th.vrem.vv v4, v8, v12, v0.t
# CHECK-INST: th.vrem.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x8c]

th.vrem.vx v4, v8, a1, v0.t
# CHECK-INST: th.vrem.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x8c]

th.vmerge.vvm v4, v8, v12, v0
# CHECK-INST: th.vmerge.vvm	v4, v8, v12, v0
# CHECK-ENCODING: [0x57,0x02,0x86,0x5c]

th.vmerge.vxm v4, v8, a1, v0
# CHECK-INST: th.vmerge.vxm	v4, v8, a1, v0
# CHECK-ENCODING: [0x57,0xc2,0x85,0x5c]

th.vmerge.vim v4, v8, 15, v0
# CHECK-INST: th.vmerge.vim	v4, v8, 15, v0
# CHECK-ENCODING: [0x57,0xb2,0x87,0x5c]

th.vmerge.vim v4, v8, -16, v0
# CHECK-INST: th.vmerge.vim	v4, v8, -16, v0
# CHECK-ENCODING: [0x57,0x32,0x88,0x5c]

th.vmv.v.v v8, v12
# CHECK-INST: th.vmv.v.v	v8, v12
# CHECK-ENCODING: [0x57,0x04,0x06,0x5e]

th.vmv.v.x v8, a1
# CHECK-INST: th.vmv.v.x	v8, a1
# CHECK-ENCODING: [0x57,0xc4,0x05,0x5e]

th.vmv.v.i v8, 15
# CHECK-INST: th.vmv.v.i	v8, 15
# CHECK-ENCODING: [0x57,0xb4,0x07,0x5e]

th.vmv.v.i v8, -16
# CHECK-INST: th.vmv.v.i	v8, -16
# CHECK-ENCODING: [0x57,0x34,0x08,0x5e]

th.vsaddu.vv v4, v8, v12
# CHECK-INST: th.vsaddu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x82]

th.vsaddu.vx v4, v8, a1
# CHECK-INST: th.vsaddu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x82]

th.vsaddu.vi v4, v8, 15
# CHECK-INST: th.vsaddu.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x82]

th.vsaddu.vi v4, v8, -16
# CHECK-INST: th.vsaddu.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x82]

th.vsaddu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsaddu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x80]

th.vsaddu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsaddu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x80]

th.vsaddu.vi v4, v8, 15, v0.t
# CHECK-INST: th.vsaddu.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x80]

th.vsaddu.vi v4, v8, -16, v0.t
# CHECK-INST: th.vsaddu.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x80]

th.vsadd.vv v4, v8, v12
# CHECK-INST: th.vsadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x86]

th.vsadd.vx v4, v8, a1
# CHECK-INST: th.vsadd.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x86]

th.vsadd.vi v4, v8, 15
# CHECK-INST: th.vsadd.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x86]

th.vsadd.vi v4, v8, -16
# CHECK-INST: th.vsadd.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x86]

th.vsadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x84]

th.vsadd.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsadd.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x84]

th.vsadd.vi v4, v8, 15, v0.t
# CHECK-INST: th.vsadd.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x84]

th.vsadd.vi v4, v8, -16, v0.t
# CHECK-INST: th.vsadd.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x84]

th.vssubu.vv v4, v8, v12
# CHECK-INST: th.vssubu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x8a]

th.vssubu.vx v4, v8, a1
# CHECK-INST: th.vssubu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x8a]

th.vssubu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vssubu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x88]

th.vssubu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vssubu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x88]

th.vssub.vv v4, v8, v12
# CHECK-INST: th.vssub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x8e]

th.vssub.vx v4, v8, a1
# CHECK-INST: th.vssub.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x8e]

th.vssub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vssub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x8c]

th.vssub.vx v4, v8, a1, v0.t
# CHECK-INST: th.vssub.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x8c]

th.vaadd.vv v4, v8, v12
# CHECK-INST: th.vaadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x92]

th.vaadd.vx v4, v8, a1
# CHECK-INST: th.vaadd.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x92]

th.vaadd.vi v4, v8, 15
# CHECK-INST: th.vaadd.vi	v4, v8, 15
# CHECK-ENCODING: [0x57,0xb2,0x87,0x92]

th.vaadd.vi v4, v8, -16
# CHECK-INST: th.vaadd.vi	v4, v8, -16
# CHECK-ENCODING: [0x57,0x32,0x88,0x92]

th.vaadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vaadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x90]

th.vaadd.vx v4, v8, a1, v0.t
# CHECK-INST: th.vaadd.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x90]

th.vaadd.vi v4, v8, 15, v0.t
# CHECK-INST: th.vaadd.vi	v4, v8, 15, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x87,0x90]

th.vaadd.vi v4, v8, -16, v0.t
# CHECK-INST: th.vaadd.vi	v4, v8, -16, v0.t
# CHECK-ENCODING: [0x57,0x32,0x88,0x90]

th.vasub.vv v4, v8, v12
# CHECK-INST: th.vasub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x9a]

th.vasub.vx v4, v8, a1
# CHECK-INST: th.vasub.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x9a]

th.vasub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vasub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x98]

th.vasub.vx v4, v8, a1, v0.t
# CHECK-INST: th.vasub.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x98]

th.vsmul.vv v4, v8, v12
# CHECK-INST: th.vsmul.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x9e]

th.vsmul.vx v4, v8, a1
# CHECK-INST: th.vsmul.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x9e]

th.vsmul.vv v4, v8, v12, v0.t
# CHECK-INST: th.vsmul.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x9c]

th.vsmul.vx v4, v8, a1, v0.t
# CHECK-INST: th.vsmul.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x9c]

th.vwsmaccu.vv v4, v12, v8
# CHECK-INST: th.vwsmaccu.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0x86,0xf2]

th.vwsmaccu.vx v4, a1, v8
# CHECK-INST: th.vwsmaccu.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xc2,0x85,0xf2]

th.vwsmacc.vv v4, v12, v8
# CHECK-INST: th.vwsmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0x86,0xf6]

th.vwsmacc.vx v4, a1, v8
# CHECK-INST: th.vwsmacc.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xc2,0x85,0xf6]

th.vwsmaccsu.vv v4, v12, v8
# CHECK-INST: th.vwsmaccsu.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x02,0x86,0xfa]

th.vwsmaccsu.vx v4, a1, v8
# CHECK-INST: th.vwsmaccsu.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xc2,0x85,0xfa]

th.vwsmaccus.vx v4, a1, v8
# CHECK-INST: th.vwsmaccus.vx	v4, a1, v8
# CHECK-ENCODING: [0x57,0xc2,0x85,0xfe]

th.vwsmaccu.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwsmaccu.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xf0]

th.vwsmaccu.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwsmaccu.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xf0]

th.vwsmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwsmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xf4]

th.vwsmacc.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwsmacc.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xf4]

th.vwsmaccsu.vv v4, v12, v8, v0.t
# CHECK-INST: th.vwsmaccsu.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xf8]

th.vwsmaccsu.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwsmaccsu.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xf8]

th.vwsmaccus.vx v4, a1, v8, v0.t
# CHECK-INST: th.vwsmaccus.vx	v4, a1, v8, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xfc]

th.vssrl.vv v4, v8, v12
# CHECK-INST: th.vssrl.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xaa]

th.vssrl.vx v4, v8, a1
# CHECK-INST: th.vssrl.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xaa]

th.vssrl.vi v4, v8, 1
# CHECK-INST: th.vssrl.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xaa]

th.vssrl.vi v4, v8, 31
# CHECK-INST: th.vssrl.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xaa]

th.vssrl.vv v4, v8, v12, v0.t
# CHECK-INST: th.vssrl.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xa8]

th.vssrl.vx v4, v8, a1, v0.t
# CHECK-INST: th.vssrl.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xa8]

th.vssrl.vi v4, v8, 1, v0.t
# CHECK-INST: th.vssrl.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xa8]

th.vssrl.vi v4, v8, 31, v0.t
# CHECK-INST: th.vssrl.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xa8]

th.vssra.vv v4, v8, v12
# CHECK-INST: th.vssra.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xae]

th.vssra.vx v4, v8, a1
# CHECK-INST: th.vssra.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xae]

th.vssra.vi v4, v8, 1
# CHECK-INST: th.vssra.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xae]

th.vssra.vi v4, v8, 31
# CHECK-INST: th.vssra.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xae]

th.vssra.vv v4, v8, v12, v0.t
# CHECK-INST: th.vssra.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xac]

th.vssra.vx v4, v8, a1, v0.t
# CHECK-INST: th.vssra.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xac]

th.vssra.vi v4, v8, 1, v0.t
# CHECK-INST: th.vssra.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xac]

th.vssra.vi v4, v8, 31, v0.t
# CHECK-INST: th.vssra.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xac]

th.vnclipu.vv v4, v8, v12
# CHECK-INST: th.vnclipu.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xba]

th.vnclipu.vx v4, v8, a1
# CHECK-INST: th.vnclipu.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xba]

th.vnclipu.vi v4, v8, 1
# CHECK-INST: th.vnclipu.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xba]

th.vnclipu.vi v4, v8, 31
# CHECK-INST: th.vnclipu.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xba]

th.vnclipu.vv v4, v8, v12, v0.t
# CHECK-INST: th.vnclipu.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xb8]

th.vnclipu.vx v4, v8, a1, v0.t
# CHECK-INST: th.vnclipu.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xb8]

th.vnclipu.vi v4, v8, 1, v0.t
# CHECK-INST: th.vnclipu.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xb8]

th.vnclipu.vi v4, v8, 31, v0.t
# CHECK-INST: th.vnclipu.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xb8]

th.vnclip.vv v4, v8, v12
# CHECK-INST: th.vnclip.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xbe]

th.vnclip.vx v4, v8, a1
# CHECK-INST: th.vnclip.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0xbe]

th.vnclip.vi v4, v8, 1
# CHECK-INST: th.vnclip.vi	v4, v8, 1
# CHECK-ENCODING: [0x57,0xb2,0x80,0xbe]

th.vnclip.vi v4, v8, 31
# CHECK-INST: th.vnclip.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xbe]

th.vnclip.vv v4, v8, v12, v0.t
# CHECK-INST: th.vnclip.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xbc]

th.vnclip.vx v4, v8, a1, v0.t
# CHECK-INST: th.vnclip.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0xbc]

th.vnclip.vi v4, v8, 1, v0.t
# CHECK-INST: th.vnclip.vi	v4, v8, 1, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x80,0xbc]

th.vnclip.vi v4, v8, 31, v0.t
# CHECK-INST: th.vnclip.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0xbc]

th.vfadd.vv v4, v8, v12
# CHECK-INST: th.vfadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x02]

th.vfadd.vf v4, v8, fa2
# CHECK-INST: th.vfadd.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x02]

th.vfadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x00]

th.vfadd.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfadd.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x00]

th.vfsub.vv v4, v8, v12
# CHECK-INST: th.vfsub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x0a]

th.vfsub.vf v4, v8, fa2
# CHECK-INST: th.vfsub.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x0a]

th.vfsub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfsub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x08]

th.vfsub.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfsub.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x08]

th.vfrsub.vf v4, v8, fa2
# CHECK-INST: th.vfrsub.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x9e]

th.vfrsub.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfrsub.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x9c]

th.vfwadd.vv v4, v8, v12
# CHECK-INST: th.vfwadd.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xc2]

th.vfwadd.vf v4, v8, fa2
# CHECK-INST: th.vfwadd.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0xc2]

th.vfwadd.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfwadd.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xc0]

th.vfwadd.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfwadd.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xc0]

th.vfwsub.vv v4, v8, v12
# CHECK-INST: th.vfwsub.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xca]

th.vfwsub.vf v4, v8, fa2
# CHECK-INST: th.vfwsub.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0xca]

th.vfwsub.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfwsub.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xc8]

th.vfwsub.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfwsub.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xc8]

th.vfwadd.wv v4, v8, v12
# CHECK-INST: th.vfwadd.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xd2]

th.vfwadd.wf v4, v8, fa2
# CHECK-INST: th.vfwadd.wf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0xd2]

th.vfwadd.wv v4, v8, v12, v0.t
# CHECK-INST: th.vfwadd.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xd0]

th.vfwadd.wf v4, v8, fa2, v0.t
# CHECK-INST: th.vfwadd.wf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xd0]

th.vfwsub.wv v4, v8, v12
# CHECK-INST: th.vfwsub.wv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xda]

th.vfwsub.wf v4, v8, fa2
# CHECK-INST: th.vfwsub.wf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0xda]

th.vfwsub.wv v4, v8, v12, v0.t
# CHECK-INST: th.vfwsub.wv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xd8]

th.vfwsub.wf v4, v8, fa2, v0.t
# CHECK-INST: th.vfwsub.wf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xd8]

th.vfmul.vv v4, v8, v12
# CHECK-INST: th.vfmul.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x92]

th.vfmul.vf v4, v8, fa2
# CHECK-INST: th.vfmul.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x92]

th.vfmul.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfmul.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x90]

th.vfmul.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfmul.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x90]

th.vfdiv.vv v4, v8, v12
# CHECK-INST: th.vfdiv.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x82]

th.vfdiv.vf v4, v8, fa2
# CHECK-INST: th.vfdiv.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x82]

th.vfdiv.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfdiv.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x80]

th.vfdiv.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfdiv.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x80]

th.vfrdiv.vf v4, v8, fa2
# CHECK-INST: th.vfrdiv.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x86]

th.vfrdiv.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfrdiv.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x84]

th.vfwmul.vv v4, v8, v12
# CHECK-INST: th.vfwmul.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xe2]

th.vfwmul.vf v4, v8, fa2
# CHECK-INST: th.vfwmul.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0xe2]

th.vfwmul.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfwmul.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xe0]

th.vfwmul.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfwmul.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xe0]

th.vfmadd.vv v4, v12, v8
# CHECK-INST: th.vfmadd.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xa2]

th.vfmadd.vf v4, fa2, v8
# CHECK-INST: th.vfmadd.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xa2]

th.vfnmadd.vv v4, v12, v8
# CHECK-INST: th.vfnmadd.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xa6]

th.vfnmadd.vf v4, fa2, v8
# CHECK-INST: th.vfnmadd.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xa6]

th.vfmsub.vv v4, v12, v8
# CHECK-INST: th.vfmsub.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xaa]

th.vfmsub.vf v4, fa2, v8
# CHECK-INST: th.vfmsub.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xaa]

th.vfnmsub.vv v4, v12, v8
# CHECK-INST: th.vfnmsub.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xae]

th.vfnmsub.vf v4, fa2, v8
# CHECK-INST: th.vfnmsub.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xae]

th.vfmadd.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfmadd.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xa0]

th.vfmadd.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfmadd.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xa0]

th.vfnmadd.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfnmadd.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xa4]

th.vfnmadd.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfnmadd.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xa4]

th.vfmsub.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfmsub.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xa8]

th.vfmsub.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfmsub.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xa8]

th.vfnmsub.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfnmsub.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xac]

th.vfnmsub.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfnmsub.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xac]

th.vfmacc.vv v4, v12, v8
# CHECK-INST: th.vfmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xb2]

th.vfmacc.vf v4, fa2, v8
# CHECK-INST: th.vfmacc.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xb2]

th.vfnmacc.vv v4, v12, v8
# CHECK-INST: th.vfnmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xb6]

th.vfnmacc.vf v4, fa2, v8
# CHECK-INST: th.vfnmacc.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xb6]

th.vfmsac.vv v4, v12, v8
# CHECK-INST: th.vfmsac.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xba]

th.vfmsac.vf v4, fa2, v8
# CHECK-INST: th.vfmsac.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xba]

th.vfnmsac.vv v4, v12, v8
# CHECK-INST: th.vfnmsac.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xbe]

th.vfnmsac.vf v4, fa2, v8
# CHECK-INST: th.vfnmsac.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xbe]

th.vfmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xb0]

th.vfmacc.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfmacc.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xb0]

th.vfnmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfnmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xb4]

th.vfnmacc.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfnmacc.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xb4]

th.vfmsac.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfmsac.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xb8]

th.vfmsac.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfmsac.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xb8]

th.vfnmsac.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfnmsac.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xbc]

th.vfnmsac.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfnmsac.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xbc]

th.vfwmacc.vv v4, v12, v8
# CHECK-INST: th.vfwmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xf2]

th.vfwmacc.vf v4, fa2, v8
# CHECK-INST: th.vfwmacc.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xf2]

th.vfwnmacc.vv v4, v12, v8
# CHECK-INST: th.vfwnmacc.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xf6]

th.vfwnmacc.vf v4, fa2, v8
# CHECK-INST: th.vfwnmacc.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xf6]

th.vfwmsac.vv v4, v12, v8
# CHECK-INST: th.vfwmsac.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xfa]

th.vfwmsac.vf v4, fa2, v8
# CHECK-INST: th.vfwmsac.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xfa]

th.vfwnmsac.vv v4, v12, v8
# CHECK-INST: th.vfwnmsac.vv	v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0xfe]

th.vfwnmsac.vf v4, fa2, v8
# CHECK-INST: th.vfwnmsac.vf	v4, fa2, v8
# CHECK-ENCODING: [0x57,0x52,0x86,0xfe]

th.vfwmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfwmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xf0]

th.vfwmacc.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfwmacc.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xf0]

th.vfwnmacc.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfwnmacc.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xf4]

th.vfwnmacc.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfwnmacc.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xf4]

th.vfwmsac.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfwmsac.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xf8]

th.vfwmsac.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfwmsac.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xf8]

th.vfwnmsac.vv v4, v12, v8, v0.t
# CHECK-INST: th.vfwnmsac.vv	v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xfc]

th.vfwnmsac.vf v4, fa2, v8, v0.t
# CHECK-INST: th.vfwnmsac.vf	v4, fa2, v8, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0xfc]

th.vfsqrt.v v4, v8
# CHECK-INST: th.vfsqrt.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x80,0x8e]

th.vfsqrt.v v4, v8, v0.t
# CHECK-INST: th.vfsqrt.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x80,0x8c]

th.vfmin.vv v4, v8, v12
# CHECK-INST: th.vfmin.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x12]

th.vfmin.vf v4, v8, fa2
# CHECK-INST: th.vfmin.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x12]

th.vfmax.vv v4, v8, v12
# CHECK-INST: th.vfmax.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x1a]

th.vfmax.vf v4, v8, fa2
# CHECK-INST: th.vfmax.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x1a]

th.vfmin.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfmin.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x10]

th.vfmin.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfmin.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x10]

th.vfmax.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfmax.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x18]

th.vfmax.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfmax.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x18]

th.vfsgnj.vv v4, v8, v12
# CHECK-INST: th.vfsgnj.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x22]

th.vfsgnj.vf v4, v8, fa2
# CHECK-INST: th.vfsgnj.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x22]

th.vfsgnjn.vv v4, v8, v12
# CHECK-INST: th.vfsgnjn.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x26]

th.vfsgnjn.vf v4, v8, fa2
# CHECK-INST: th.vfsgnjn.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x26]

th.vfsgnjx.vv v4, v8, v12
# CHECK-INST: th.vfsgnjx.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x2a]

th.vfsgnjx.vf v4, v8, fa2
# CHECK-INST: th.vfsgnjx.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x2a]

th.vfsgnj.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfsgnj.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x20]

th.vfsgnj.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfsgnj.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x20]

th.vfsgnjn.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfsgnjn.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x24]

th.vfsgnjn.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfsgnjn.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x24]

th.vfsgnjx.vv v4, v8, v12, v0.t
# CHECK-INST: th.vfsgnjx.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x28]

th.vfsgnjx.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vfsgnjx.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x28]

th.vmfgt.vv v4, v8, v12
# CHECK-INST: th.vmflt.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0xc4,0x6e]

th.vmfge.vv v4, v8, v12
# CHECK-INST: th.vmfle.vv v4, v12, v8
# CHECK-ENCODING: [0x57,0x12,0xc4,0x66]

th.vmfgt.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmflt.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0xc4,0x6c]

th.vmfge.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmfle.vv v4, v12, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0xc4,0x64]

th.vmfeq.vv v4, v8, v12
# CHECK-INST: th.vmfeq.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x62]

th.vmfeq.vf v4, v8, fa2
# CHECK-INST: th.vmfeq.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x62]

th.vmfne.vv v4, v8, v12
# CHECK-INST: th.vmfne.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x72]

th.vmfne.vf v4, v8, fa2
# CHECK-INST: th.vmfne.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x72]

th.vmflt.vv v4, v8, v12
# CHECK-INST: th.vmflt.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x6e]

th.vmflt.vf v4, v8, fa2
# CHECK-INST: th.vmflt.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x6e]

th.vmfle.vv v4, v8, v12
# CHECK-INST: th.vmfle.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x66]

th.vmfle.vf v4, v8, fa2
# CHECK-INST: th.vmfle.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x66]

th.vmfgt.vf v4, v8, fa2
# CHECK-INST: th.vmfgt.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x76]

th.vmfge.vf v4, v8, fa2
# CHECK-INST: th.vmfge.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x7e]

th.vmfeq.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmfeq.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x60]

th.vmfeq.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmfeq.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x60]

th.vmfne.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmfne.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x70]

th.vmfne.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmfne.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x70]

th.vmflt.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmflt.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x6c]

th.vmflt.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmflt.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x6c]

th.vmfle.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmfle.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x64]

th.vmfle.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmfle.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x64]

th.vmfgt.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmfgt.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x74]

th.vmfge.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmfge.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x7c]

th.vmford.vv v4, v8, v12
# CHECK-INST: th.vmford.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x6a]

th.vmford.vf v4, v8, fa2
# CHECK-INST: th.vmford.vf	v4, v8, fa2
# CHECK-ENCODING: [0x57,0x52,0x86,0x6a]

th.vmford.vv v4, v8, v12, v0.t
# CHECK-INST: th.vmford.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x68]

th.vmford.vf v4, v8, fa2, v0.t
# CHECK-INST: th.vmford.vf	v4, v8, fa2, v0.t
# CHECK-ENCODING: [0x57,0x52,0x86,0x68]

th.vfclass.v v4, v8
# CHECK-INST: th.vfclass.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x88,0x8e]

th.vfclass.v v4, v8, v0.t
# CHECK-INST: th.vfclass.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x88,0x8c]

th.vfmerge.vfm v4, v8, fa2, v0
# CHECK-INST: th.vfmerge.vfm	v4, v8, fa2, v0
# CHECK-ENCODING: [0x57,0x52,0x86,0x5c]

th.vfmv.v.f v4, fa1
# CHECK-INST: th.vfmv.v.f	v4, fa1
# CHECK-ENCODING: [0x57,0xd2,0x05,0x5e]

th.vfcvt.xu.f.v v4, v8
# CHECK-INST: th.vfcvt.xu.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x80,0x8a]

th.vfcvt.x.f.v v4, v8
# CHECK-INST: th.vfcvt.x.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x80,0x8a]

th.vfcvt.f.xu.v v4, v8
# CHECK-INST: th.vfcvt.f.xu.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x81,0x8a]

th.vfcvt.f.x.v v4, v8
# CHECK-INST: th.vfcvt.f.x.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x81,0x8a]

th.vfcvt.xu.f.v v4, v8, v0.t
# CHECK-INST: th.vfcvt.xu.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x80,0x88]

th.vfcvt.x.f.v v4, v8, v0.t
# CHECK-INST: th.vfcvt.x.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x80,0x88]

th.vfcvt.f.xu.v v4, v8, v0.t
# CHECK-INST: th.vfcvt.f.xu.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x81,0x88]

th.vfcvt.f.x.v v4, v8, v0.t
# CHECK-INST: th.vfcvt.f.x.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x81,0x88]

th.vfwcvt.xu.f.v v4, v8
# CHECK-INST: th.vfwcvt.xu.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x84,0x8a]

th.vfwcvt.x.f.v v4, v8
# CHECK-INST: th.vfwcvt.x.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x84,0x8a]

th.vfwcvt.f.xu.v v4, v8
# CHECK-INST: th.vfwcvt.f.xu.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x85,0x8a]

th.vfwcvt.f.x.v v4, v8
# CHECK-INST: th.vfwcvt.f.x.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x85,0x8a]

th.vfwcvt.f.f.v v4, v8
# CHECK-INST: th.vfwcvt.f.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x86,0x8a]

th.vfwcvt.xu.f.v v4, v8, v0.t
# CHECK-INST: th.vfwcvt.xu.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x84,0x88]

th.vfwcvt.x.f.v v4, v8, v0.t
# CHECK-INST: th.vfwcvt.x.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x84,0x88]

th.vfwcvt.f.xu.v v4, v8, v0.t
# CHECK-INST: th.vfwcvt.f.xu.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x85,0x88]

th.vfwcvt.f.x.v v4, v8, v0.t
# CHECK-INST: th.vfwcvt.f.x.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x85,0x88]

th.vfwcvt.f.f.v v4, v8, v0.t
# CHECK-INST: th.vfwcvt.f.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x88]

th.vfncvt.xu.f.v v4, v8
# CHECK-INST: th.vfncvt.xu.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x88,0x8a]

th.vfncvt.x.f.v v4, v8
# CHECK-INST: th.vfncvt.x.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x88,0x8a]

th.vfncvt.f.xu.v v4, v8
# CHECK-INST: th.vfncvt.f.xu.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x89,0x8a]

th.vfncvt.f.x.v v4, v8
# CHECK-INST: th.vfncvt.f.x.v	v4, v8
# CHECK-ENCODING: [0x57,0x92,0x89,0x8a]

th.vfncvt.f.f.v v4, v8
# CHECK-INST: th.vfncvt.f.f.v	v4, v8
# CHECK-ENCODING: [0x57,0x12,0x8a,0x8a]

th.vfncvt.xu.f.v v4, v8, v0.t
# CHECK-INST: th.vfncvt.xu.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x88,0x88]

th.vfncvt.x.f.v v4, v8, v0.t
# CHECK-INST: th.vfncvt.x.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x88,0x88]

th.vfncvt.f.xu.v v4, v8, v0.t
# CHECK-INST: th.vfncvt.f.xu.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x89,0x88]

th.vfncvt.f.x.v v4, v8, v0.t
# CHECK-INST: th.vfncvt.f.x.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x92,0x89,0x88]

th.vfncvt.f.f.v v4, v8, v0.t
# CHECK-INST: th.vfncvt.f.f.v	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x12,0x8a,0x88]

th.vredsum.vs v4, v8, v12
# CHECK-INST: th.vredsum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x02]

th.vredmaxu.vs v4, v8, v8
# CHECK-INST: th.vredmaxu.vs	v4, v8, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x1a]

th.vredmax.vs v4, v8, v8
# CHECK-INST: th.vredmax.vs	v4, v8, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x1e]

th.vredminu.vs v4, v8, v8
# CHECK-INST: th.vredminu.vs	v4, v8, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x12]

th.vredmin.vs v4, v8, v8
# CHECK-INST: th.vredmin.vs	v4, v8, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x16]

th.vredand.vs v4, v8, v12
# CHECK-INST: th.vredand.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x06]

th.vredor.vs v4, v8, v12
# CHECK-INST: th.vredor.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x0a]

th.vredxor.vs v4, v8, v12
# CHECK-INST: th.vredxor.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x0e]

th.vredsum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vredsum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x00]

th.vredmaxu.vs v4, v8, v8, v0.t
# CHECK-INST: th.vredmaxu.vs	v4, v8, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x84,0x18]

th.vredmax.vs v4, v8, v8, v0.t
# CHECK-INST: th.vredmax.vs	v4, v8, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x84,0x1c]

th.vredminu.vs v4, v8, v8, v0.t
# CHECK-INST: th.vredminu.vs	v4, v8, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x84,0x10]

th.vredmin.vs v4, v8, v8, v0.t
# CHECK-INST: th.vredmin.vs	v4, v8, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x84,0x14]

th.vredand.vs v4, v8, v12, v0.t
# CHECK-INST: th.vredand.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x04]

th.vredor.vs v4, v8, v12, v0.t
# CHECK-INST: th.vredor.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x08]

th.vredxor.vs v4, v8, v12, v0.t
# CHECK-INST: th.vredxor.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x22,0x86,0x0c]

th.vwredsumu.vs v4, v8, v12
# CHECK-INST: th.vwredsumu.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xc2]

th.vwredsum.vs v4, v8, v12
# CHECK-INST: th.vwredsum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0xc6]

th.vwredsumu.vs v4, v8, v12, v0.t
# CHECK-INST: th.vwredsumu.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xc0]

th.vwredsum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vwredsum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0xc4]

th.vfredosum.vs v4, v8, v12
# CHECK-INST: th.vfredosum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x0e]

th.vfredsum.vs v4, v8, v12
# CHECK-INST: th.vfredsum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x06]

th.vfredmax.vs v4, v8, v12
# CHECK-INST: th.vfredmax.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x1e]

th.vfredmin.vs v4, v8, v12
# CHECK-INST: th.vfredmin.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0x16]

th.vfredosum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfredosum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x0c]

th.vfredsum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfredsum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x04]

th.vfredmax.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfredmax.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x1c]

th.vfredmin.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfredmin.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0x14]

th.vfwredosum.vs v4, v8, v12
# CHECK-INST: th.vfwredosum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xce]

th.vfwredsum.vs v4, v8, v12
# CHECK-INST: th.vfwredsum.vs	v4, v8, v12
# CHECK-ENCODING: [0x57,0x12,0x86,0xc6]

th.vfwredosum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfwredosum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xcc]

th.vfwredsum.vs v4, v8, v12, v0.t
# CHECK-INST: th.vfwredsum.vs	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x12,0x86,0xc4]

th.vmcpy.m v4, v8
# CHECK-INST: th.vmcpy.m v4, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x66]

th.vmclr.m v4
# CHECK-INST: th.vmclr.m v4
# CHECK-ENCODING: [0x57,0x22,0x42,0x6e]

th.vmset.m v4
# CHECK-INST: th.vmset.m v4
# CHECK-ENCODING: [0x57,0x22,0x42,0x7e]

th.vmnot.m v4, v8
# CHECK-INST: th.vmnot.m v4, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x76]

th.vmand.mm v4, v8, v12
# CHECK-INST: th.vmand.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x66]

th.vmnand.mm v4, v8, v12
# CHECK-INST: th.vmnand.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x76]

th.vmandnot.mm v4, v8, v12
# CHECK-INST: th.vmandnot.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x62]

th.vmxor.mm v4, v8, v12
# CHECK-INST: th.vmxor.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x6e]

th.vmor.mm v4, v8, v12
# CHECK-INST: th.vmor.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x6a]

th.vmnor.mm v4, v8, v12
# CHECK-INST: th.vmnor.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x7a]

th.vmornot.mm v4, v8, v12
# CHECK-INST: th.vmornot.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x72]

th.vmxnor.mm v4, v8, v12
# CHECK-INST: th.vmxnor.mm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x7e]

th.vmpopc.m a0, v12
# CHECK-INST: th.vmpopc.m	a0, v12
# CHECK-ENCODING: [0x57,0x25,0xc0,0x52]

th.vmfirst.m a0, v12
# CHECK-INST: th.vmfirst.m	a0, v12
# CHECK-ENCODING: [0x57,0x25,0xc0,0x56]

th.vmsbf.m v4, v8
# CHECK-INST: th.vmsbf.m	v4, v8
# CHECK-ENCODING: [0x57,0xa2,0x80,0x5a]

th.vmsif.m v4, v8
# CHECK-INST: th.vmsif.m	v4, v8
# CHECK-ENCODING: [0x57,0xa2,0x81,0x5a]

th.vmsof.m v4, v8
# CHECK-INST: th.vmsof.m	v4, v8
# CHECK-ENCODING: [0x57,0x22,0x81,0x5a]

th.viota.m v4, v8
# CHECK-INST: th.viota.m	v4, v8
# CHECK-ENCODING: [0x57,0x22,0x88,0x5a]

th.vid.v v4
# CHECK-INST: th.vid.v	v4
# CHECK-ENCODING: [0x57,0xa2,0x08,0x5a]

th.vmpopc.m a0, v12, v0.t
# CHECK-INST: th.vmpopc.m	a0, v12, v0.t
# CHECK-ENCODING: [0x57,0x25,0xc0,0x50]

th.vmfirst.m a0, v12, v0.t
# CHECK-INST: th.vmfirst.m	a0, v12, v0.t
# CHECK-ENCODING: [0x57,0x25,0xc0,0x54]

th.vmsbf.m v4, v8, v0.t
# CHECK-INST: th.vmsbf.m	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0xa2,0x80,0x58]

th.vmsif.m v4, v8, v0.t
# CHECK-INST: th.vmsif.m	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0xa2,0x81,0x58]

th.vmsof.m v4, v8, v0.t
# CHECK-INST: th.vmsof.m	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x81,0x58]

th.viota.m v4, v8, v0.t
# CHECK-INST: th.viota.m	v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x22,0x88,0x58]

th.vid.v v4, v0.t
# CHECK-INST: th.vid.v	v4, v0.t
# CHECK-ENCODING: [0x57,0xa2,0x08,0x58]

th.vmv.x.s a0, v12
# CHECK-INST: th.vmv.x.s a0, v12
# CHECK-ENCODING: [0x57,0x25,0xc0,0x32]

th.vext.x.v a0, v12, a2
# CHECK-INST: th.vext.x.v	a0, v12, a2
# CHECK-ENCODING: [0x57,0x25,0xc6,0x32]

th.vmv.s.x v4, a0
# CHECK-INST: th.vmv.s.x	v4, a0
# CHECK-ENCODING: [0x57,0x62,0x05,0x36]

th.vfmv.f.s fa0, v8
# CHECK-INST: th.vfmv.f.s	fa0, v8
# CHECK-ENCODING: [0x57,0x15,0x80,0x32]

th.vfmv.s.f v4, fa1
# CHECK-INST: th.vfmv.s.f	v4, fa1
# CHECK-ENCODING: [0x57,0xd2,0x05,0x36]

th.vslideup.vx v4, v8, a1
# CHECK-INST: th.vslideup.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x3a]

th.vslideup.vi v4, v8, 0
# CHECK-INST: th.vslideup.vi	v4, v8, 0
# CHECK-ENCODING: [0x57,0x32,0x80,0x3a]

th.vslideup.vi v4, v8, 31
# CHECK-INST: th.vslideup.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x3a]

th.vslidedown.vx v4, v8, a1
# CHECK-INST: th.vslidedown.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x3e]

th.vslidedown.vi v4, v8, 0
# CHECK-INST: th.vslidedown.vi	v4, v8, 0
# CHECK-ENCODING: [0x57,0x32,0x80,0x3e]

th.vslidedown.vi v4, v8, 31
# CHECK-INST: th.vslidedown.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x3e]

th.vslideup.vx v4, v8, a1, v0.t
# CHECK-INST: th.vslideup.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x38]

th.vslideup.vi v4, v8, 0, v0.t
# CHECK-INST: th.vslideup.vi	v4, v8, 0, v0.t
# CHECK-ENCODING: [0x57,0x32,0x80,0x38]

th.vslideup.vi v4, v8, 31, v0.t
# CHECK-INST: th.vslideup.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x38]

th.vslidedown.vx v4, v8, a1, v0.t
# CHECK-INST: th.vslidedown.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x3c]

th.vslidedown.vi v4, v8, 0, v0.t
# CHECK-INST: th.vslidedown.vi	v4, v8, 0, v0.t
# CHECK-ENCODING: [0x57,0x32,0x80,0x3c]

th.vslidedown.vi v4, v8, 31, v0.t
# CHECK-INST: th.vslidedown.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x3c]

th.vslide1up.vx v4, v8, a1
# CHECK-INST: th.vslide1up.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x3a]

th.vslide1down.vx v4, v8, a1
# CHECK-INST: th.vslide1down.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xe2,0x85,0x3e]

th.vslide1up.vx v4, v8, a1, v0.t
# CHECK-INST: th.vslide1up.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x38]

th.vslide1down.vx v4, v8, a1, v0.t
# CHECK-INST: th.vslide1down.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xe2,0x85,0x3c]

th.vrgather.vv v4, v8, v12
# CHECK-INST: th.vrgather.vv	v4, v8, v12
# CHECK-ENCODING: [0x57,0x02,0x86,0x32]

th.vrgather.vx v4, v8, a1
# CHECK-INST: th.vrgather.vx	v4, v8, a1
# CHECK-ENCODING: [0x57,0xc2,0x85,0x32]

th.vrgather.vi v4, v8, 0
# CHECK-INST: th.vrgather.vi	v4, v8, 0
# CHECK-ENCODING: [0x57,0x32,0x80,0x32]

th.vrgather.vi v4, v8, 31
# CHECK-INST: th.vrgather.vi	v4, v8, 31
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x32]

th.vrgather.vv v4, v8, v12, v0.t
# CHECK-INST: th.vrgather.vv	v4, v8, v12, v0.t
# CHECK-ENCODING: [0x57,0x02,0x86,0x30]

th.vrgather.vx v4, v8, a1, v0.t
# CHECK-INST: th.vrgather.vx	v4, v8, a1, v0.t
# CHECK-ENCODING: [0x57,0xc2,0x85,0x30]

th.vrgather.vi v4, v8, 0, v0.t
# CHECK-INST: th.vrgather.vi	v4, v8, 0, v0.t
# CHECK-ENCODING: [0x57,0x32,0x80,0x30]

th.vrgather.vi v4, v8, 31, v0.t
# CHECK-INST: th.vrgather.vi	v4, v8, 31, v0.t
# CHECK-ENCODING: [0x57,0xb2,0x8f,0x30]

th.vcompress.vm v4, v8, v12
# CHECK-INST: th.vcompress.vm	v4, v8, v12
# CHECK-ENCODING: [0x57,0x22,0x86,0x5e]

csrr a0, vstart
# CHECK-INST: csrr	a0, vstart
# CHECK-ENCODING: [0x73,0x25,0x80,0x00]

csrr a0, vxsat
# CHECK-INST: csrr	a0, vxsat
# CHECK-ENCODING: [0x73,0x25,0x90,0x00]

csrr a0, vxrm
# CHECK-INST: csrr	a0, vxrm
# CHECK-ENCODING: [0x73,0x25,0xa0,0x00]

csrr a0, vl
# CHECK-INST: csrr	a0, vl
# CHECK-ENCODING: [0x73,0x25,0x00,0xc2]

csrr a0, vtype
# CHECK-INST: csrr	a0, vtype
# CHECK-ENCODING: [0x73,0x25,0x10,0xc2]
