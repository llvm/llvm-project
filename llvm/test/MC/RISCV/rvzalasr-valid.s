# RUN: llvm-mc %s -triple=riscv32 -mattr=+zalasr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zalasr < %s \
# RUN:     | llvm-objdump --mattr=+zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zalasr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zalasr < %s \
# RUN:     | llvm-objdump --mattr=+zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: lb.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x03,0x05,0x34]
lb.aq t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: lh.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x13,0x05,0x34]
lh.aq t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: lw.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x23,0x05,0x34]
lw.aq t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: lb.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x03,0x05,0x36]
lb.aqrl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: lh.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x13,0x05,0x36]
lh.aqrl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: lw.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x23,0x05,0x36]
lw.aqrl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}


# CHECK-ASM-AND-OBJ: sb.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x00,0x65,0x3a]
sb.rl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sh.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x10,0x65,0x3a]
sh.rl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sw.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x20,0x65,0x3a]
sw.rl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sb.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x00,0x65,0x3e]
sb.aqrl t1, (a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sh.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x10,0x65,0x3e]
sh.aqrl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}

# CHECK-ASM-AND-OBJ: sw.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x20,0x65,0x3e]
sw.aqrl t1, 0(a0) # expected-error {{instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions)}}
