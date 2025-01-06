# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfbfmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfbfmin -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfbfmin,+f < %s \
# RUN:     | llvm-objdump --mattr=+zfbfmin --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfbfmin,+f < %s \
# RUN:     | llvm-objdump --mattr=+zfbfmin --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: flh ft0, 12(a0)
# CHECK-ASM: encoding: [0x07,0x10,0xc5,0x00]
flh f0, 12(a0)
# CHECK-ASM-AND-OBJ: flh ft1, 4(ra)
# CHECK-ASM: encoding: [0x87,0x90,0x40,0x00]
flh f1, +4(ra)
# CHECK-ASM-AND-OBJ: flh ft2, -2048(a3)
# CHECK-ASM: encoding: [0x07,0x91,0x06,0x80]
flh f2, -2048(x13)
# CHECK-ASM-AND-OBJ: flh ft3, -2048(s1)
# CHECK-ASM: encoding: [0x87,0x91,0x04,0x80]
flh f3, %lo(2048)(s1)
# CHECK-ASM-AND-OBJ: flh ft4, 2047(s2)
# CHECK-ASM: encoding: [0x07,0x12,0xf9,0x7f]
flh f4, 2047(s2)
# CHECK-ASM-AND-OBJ: flh ft5, 0(s3)
# CHECK-ASM: encoding: [0x87,0x92,0x09,0x00]
flh f5, 0(s3)

# CHECK-ASM-AND-OBJ: fsh ft6, 2047(s4)
# CHECK-ASM: encoding: [0xa7,0x1f,0x6a,0x7e]
fsh f6, 2047(s4)
# CHECK-ASM-AND-OBJ: fsh ft7, -2048(s5)
# CHECK-ASM: encoding: [0x27,0x90,0x7a,0x80]
fsh f7, -2048(s5)
# CHECK-ASM-AND-OBJ: fsh fs0, -2048(s6)
# CHECK-ASM: encoding: [0x27,0x10,0x8b,0x80]
fsh f8, %lo(2048)(s6)
# CHECK-ASM-AND-OBJ: fsh fs1, 999(s7)
# CHECK-ASM: encoding: [0xa7,0x93,0x9b,0x3e]
fsh f9, 999(s7)

# CHECK-ASM-AND-OBJ: fmv.x.h a2, fs7
# CHECK-ASM: encoding: [0x53,0x86,0x0b,0xe4]
fmv.x.h a2, fs7
# CHECK-ASM-AND-OBJ: fmv.h.x ft1, a6
# CHECK-ASM: encoding: [0xd3,0x00,0x08,0xf4]
fmv.h.x ft1, a6

# CHECK-ASM-AND-OBJ: fcvt.s.bf16 fa0, ft0
# CHECK-ASM: encoding: [0x53,0x05,0x60,0x40]
fcvt.s.bf16 fa0, ft0
# CHECK-ASM-AND-OBJ: fcvt.s.bf16 fa0, ft0, rup
# CHECK-ASM: encoding: [0x53,0x35,0x60,0x40]
fcvt.s.bf16 fa0, ft0, rup
# CHECK-ASM-AND-OBJ: fcvt.bf16.s ft2, fa2
# CHECK-ASM: encoding: [0x53,0x71,0x86,0x44]
fcvt.bf16.s ft2, fa2
