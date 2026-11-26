# RUN: llvm-mc %s -triple=riscv64 -mattr=+zalasr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zalasr < %s \
# RUN:     | llvm-objdump --mattr=+zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s


# CHECK-ASM-AND-OBJ: ld.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x33,0x05,0x34]
ld.aq t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: ld.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x33,0x05,0x36]
ld.aqrl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}


# CHECK-ASM-AND-OBJ: sd.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x30,0x65,0x3a]
sd.rl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sd.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x30,0x65,0x3e]
sd.aqrl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}
