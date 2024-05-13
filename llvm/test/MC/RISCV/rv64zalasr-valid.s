# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zalasr -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zalasr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zalasr -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefixes=CHECK-NO-EXT %s


# CHECK-ASM-AND-OBJ: ld.aq t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x33,0x05,0x34]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
ld.aq t1, (a0)

# CHECK-ASM-AND-OBJ: ld.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x33,0x05,0x36]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
ld.aqrl t1, 0(a0)


# CHECK-ASM-AND-OBJ: sd.rl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x30,0x65,0x3a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sd.rl t1, 0(a0)

# CHECK-ASM-AND-OBJ: sd.aqrl t1, (a0)
# CHECK-ASM: encoding: [0x2f,0x30,0x65,0x3e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zalasr' (Load-Acquire and Store-Release Instructions){{$}}
sd.aqrl t1, (a0)
