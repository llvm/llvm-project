# RUN: llvm-mc --triple=riscv32 --mattr=+experimental-y,+experimental-zyhybrid --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-ASM-AND-OBJ
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+experimental-zyhybrid --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-ASM-AND-OBJ
## Check that we get the same results in RVI hybrid mode:
# RUN: llvm-mc --triple=riscv32 --mattr=-experimental-y,+experimental-zyhybrid --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-ASM-AND-OBJ
# RUN: llvm-mc --triple=riscv64 --mattr=-experimental-y,+experimental-zyhybrid --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-ASM-AND-OBJ
## Check disassembly for all four combinations:
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y,+experimental-zyhybrid --riscv-add-build-attributes < %s \
# RUN:   | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y,+experimental-zyhybrid --riscv-add-build-attributes < %s \
# RUN:   | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=-experimental-y,+experimental-zyhybrid --riscv-add-build-attributes < %s \
# RUN:   | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=-experimental-y,+experimental-zyhybrid --riscv-add-build-attributes < %s \
# RUN:   | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ

ymoder a0, a1
# CHECK-ASM-AND-OBJ: ymoder	a0, a1
# CHECK-SAME: # encoding: [0x33,0x85,0x35,0x10]
# CHECK-NEXT: # <MCInst #[[#MCINST1:]] YMODER{{$}}
# CHECK-NEXT: #  <MCOperand Reg:X10>
# CHECK-NEXT: #  <MCOperand Reg:X11_Y>>
ymodew a0, a1, a2
# CHECK-ASM-AND-OBJ-NEXT: ymodew	a0, a1, a2
# CHECK-SAME: # encoding: [0x33,0xf5,0xc5,0x0c]
# CHECK-NEXT: # <MCInst #[[#MCINST2:]] YMODEW{{$}}
# CHECK-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-NEXT: #  <MCOperand Reg:X12>>
## TODO: ymodesw.y effectively sets +experimental-y, we should probably emit
## that as well, so that subsequent instructions use the expected mode.
ymodeswy
# CHECK-ASM-AND-OBJ-NEXT: ymodeswy
# CHECK-SAME:  # encoding: [0x33,0x10,0x00,0x12]
# CHECK-NEXT: # <MCInst #[[#MCINST3:]] YMODESWY>{{$}}
## TODO: ymodesw.y effectively sets -experimental-y, we should probably emit
## that as well, so that subsequent instructions use the expected mode.
ymodeswi
# CHECK-ASM-AND-OBJ-NEXT: ymodeswi
# CHECK-SAME: encoding: [0x33,0x10,0x00,0x14]
# CHECK-NEXT: # <MCInst #[[#MCINST4:]] YMODESWI>{{$}}
