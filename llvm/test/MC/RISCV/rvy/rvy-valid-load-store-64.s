// RUN: llvm-mc --triple=riscv64 --mattr=+experimental-zyhybrid --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
// RUN: llvm-mc --triple=riscv64 --mattr=-experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
// RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-zyhybrid --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s

/// Both capability & normal RISC-V instruction use the same encoding, and the
/// same MCInst as we rely on RegClassByHwMode to select the right base pointer.
/// The only difference is that using register x0 as the base is illegal for
/// RVY (see rvy-invalid-load-store.s).

lwu a0, 0(a1)
// CHECK: lwu	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x03,0xe5,0x05,0x00]
// CHECK-ASM-NEXT: # <MCInst #[[#MCINST1:]] LWU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP: #  <MCOperand Reg:X11_Y>
// CHECK-INT: #  <MCOperand Reg:X11>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
ld a0, 0(a1)
// CHECK: ld	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x03,0xb5,0x05,0x00]
// CHECK-ASM-NEXT: # <MCInst #[[#MCINST2:]] LD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP: #  <MCOperand Reg:X11_Y>
// CHECK-INT: #  <MCOperand Reg:X11>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sd a0, 0(a1)
// CHECK: sd	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x23,0xb0,0xa5,0x00]
// CHECK-ASM-NEXT: # <MCInst #[[#MCINST3:]] SD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP: #  <MCOperand Reg:X11_Y>
// CHECK-INT: #  <MCOperand Reg:X11>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
