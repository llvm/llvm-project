# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex - \
# RUN:   | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y,+cap-mode < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex - \
# RUN:   | FileCheck %s

## Both capability & normal RISC-V instruction use the same encoding, and the
## same MCInst as we rely on RegClassByHwMode to select the rigt base pointer.
# CHECK:	lwu	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0xe5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LWU{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lwu a0, 0(a1)
# CHECK-NEXT:	ld	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x03,0xb5,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] LD{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
ld a0, 0(a1)
# CHECK-NEXT:	sd	a0, 0(a1)
# CHECK-ASM-SAME: # encoding: [0x23,0xb0,0xa5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SD{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X11>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sd a0, 0(a1)


#
# TODO: Pseudos using a symbol expanding to an AUIPC pair:
#
# lwu a0, sym
# ld a0, sym
# sd a0, sym, t0
