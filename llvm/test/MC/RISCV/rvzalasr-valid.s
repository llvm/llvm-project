# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zalasr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zalasr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zalasr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zalasr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: lb.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x03,0x05,0x34]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lb.aq t1, 0(a0)

# CHECK-ASM-AND-OBJ: lh.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x13,0x05,0x34]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lh.aq t1, 0(a0)

# CHECK-ASM-AND-OBJ: lw.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x23,0x05,0x34]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lw.aq t1, (a0)

# CHECK-ASM-AND-OBJ: lb.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x03,0x05,0x36]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lb.aqrl t1, 0(a0)

# CHECK-ASM-AND-OBJ: lh.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x13,0x05,0x36]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lh.aqrl t1, (a0)

# CHECK-ASM-AND-OBJ: lw.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x23,0x05,0x36]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
lw.aqrl t1, (a0)


# CHECK-ASM-AND-OBJ: sb.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x00,0x65,0x3a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sb.rl t1, (a0)

# CHECK-ASM-AND-OBJ: sh.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x10,0x65,0x3a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sh.rl t1, 0(a0)

# CHECK-ASM-AND-OBJ: sw.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x20,0x65,0x3a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sw.rl t1, (a0)

# CHECK-ASM-AND-OBJ: sb.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x00,0x65,0x3e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sb.aqrl t1, (a0)

# CHECK-ASM-AND-OBJ: sh.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x10,0x65,0x3e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sh.aqrl t1, 0(a0)

# CHECK-ASM-AND-OBJ: sw.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x20,0x65,0x3e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sw.aqrl t1, 0(a0)
