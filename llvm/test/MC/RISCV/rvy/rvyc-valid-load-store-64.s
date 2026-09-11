// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,-experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT %s
// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+c,+zcb,-experimental-y --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+c,+zcb,+experimental-y --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s

c.ldsp a0, 16(sp)


// Note: No c.lwu instruction in zcb
// CHECK:	c.ld	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0x61]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
c.ld a0, 0(a1)
// CHECK-NEXT:	c.sd	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0xe1]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
c.sd a0, 0(a1)
// CHECK-NEXT:	c.ldsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x42,0x65]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LDSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
c.ldsp a0, 16(sp)
// CHECK-NEXT:	c.sdsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x2a,0xe8]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SDSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
c.sdsp a0, 16(sp)

/// Check that compress patterns work as expected.

// Note: No c.lwu instruction in zcb
// CHECK:	{{[[:space:]]}}lwu	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x03,0xe5,0x05,0x00]
// CHECK-ASM-NEXT: # <MCInst #[[#]] LWU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
lwu a0, 0(a1)
// CHECK-NEXT:	c.ld	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0x61]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
ld a0, 0(a1)
// CHECK-NEXT:	c.sd	a0, 0(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0xe1]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
sd a0, 0(a1)
// CHECK-NEXT:	c.ldsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x42,0x65]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LDSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
ld a0, 16(sp)
// CHECK-NEXT:	c.sdsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x2a,0xe8]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SDSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
sd a0, 16(sp)
