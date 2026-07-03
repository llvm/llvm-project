# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicbom -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zicbom -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicbom < %s \
# RUN:     | llvm-objdump --mattr=+zicbom -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zicbom < %s \
# RUN:     | llvm-objdump --mattr=+zicbom -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cbo.clean (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x12,0x00]
cbo.clean (t0)
# CHECK-ASM-AND-OBJ: cbo.clean (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x12,0x00]
cbo.clean 0(t0)

# CHECK-ASM-AND-OBJ: cbo.flush (t1)
# CHECK-ASM: encoding: [0x0f,0x20,0x23,0x00]
cbo.flush (t1)
# CHECK-ASM-AND-OBJ: cbo.flush (t1)
# CHECK-ASM: encoding: [0x0f,0x20,0x23,0x00]
cbo.flush 0(t1)

# CHECK-ASM-AND-OBJ: cbo.inval (t2)
# CHECK-ASM: encoding: [0x0f,0xa0,0x03,0x00]
cbo.inval (t2)
# CHECK-ASM-AND-OBJ: cbo.inval (t2)
# CHECK-ASM: encoding: [0x0f,0xa0,0x03,0x00]
cbo.inval 0(t2)
