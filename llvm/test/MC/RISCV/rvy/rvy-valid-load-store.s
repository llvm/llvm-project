# RUN: llvm-mc --triple=riscv32 -mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
# RUN: llvm-mc --triple=riscv32 -mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex - | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y,+cap-mode < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex - | FileCheck %s

# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex - | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y,+cap-mode < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex - | FileCheck %s

## Both capability & normal RISC-V instruction use the same encoding, and the
## same MCInst as we rely on RegClassByHwMode to select the rigt base pointer.

# CHECK: lb	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0x85,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LB{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lb a0, 0(a1)
# CHECK-NEXT: lbu	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0xc5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LBU{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lbu a0, 0(a1)
# CHECK-NEXT: lh	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0x95,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LH{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lh a0, 0(a1)
# CHECK-NEXT: lhu	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0xd5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LHU{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lhu a0, 0(a1)
# CHECK-NEXT: lw	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0xa5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LW{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lw a0, 0(a1)
# CHECK-NEXT: sb	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x23,0x80,0xa5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SB{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sb a0, 0(a1)
# CHECK-NEXT: sh	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x23,0x90,0xa5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SH{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sh a0, 0(a1)
# CHECK-NEXT: sw	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x23,0xa0,0xa5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SW{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sw a0, 0(a1)
#
## Capability load & store
#
# CHECK-NEXT: ly	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x0f,0xc5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
ly a0, 0(a1)
# CHECK-NEXT: sy	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x23,0xc0,0xa5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sy a0, 0(a1)


# TODO: Test the pseudo expansions using AUIPC:
# lb a0, sym
# lbu a0, sym
# lh a0, sym
# lhu a0, sym
# lw a0, sym
# ly a0, sym
#
# sb a0, sym, t0
# sh a0, sym, t0
# sw a0, sym, t0
# sy a0, sym, t0
