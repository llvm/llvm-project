// RUN: llvm-mc --triple=riscv32 --mattr=+c,+zcb,+rvy-int-mode --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv32 --mattr=+c,+zcb,+experimental-y --defsym=RVY=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-32,CHECK-CAP-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+c,+zcb,+rvy-int-mode --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+c,+zcb,+experimental-y --defsym=RVY=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-CAP-ASM-AND-OBJ

// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,+rvy-int-mode --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-y --defsym=RVY=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-64,CHECK-CAP-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+c,+zcb,+rvy-int-mode --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+c,+zcb,+experimental-y --defsym=RVY=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-CAP-ASM-AND-OBJ

c.lbu a0, 1(a1)
// CHECK-ASM-AND-OBJ: c.lbu	a0, 1(a1)
// CHECK-ASM-SAME: # encoding: [0xc8,0x81]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LBU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
c.lh a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lh	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xe8,0x85]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LH{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
c.lhu a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lhu	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xa8,0x85]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LHU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
c.lw a0, 16(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lw	a0, 16(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0x49]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
c.sb a0, 1(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sb	a0, 1(a1)
// CHECK-ASM-SAME: # encoding: [0xc8,0x89]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SB{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
c.sh a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sh	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xa8,0x8d]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SH{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
c.sw a0, 16(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sw	a0, 16(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0xc9]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
//
/// Compressed Capability load & store (only in RVY mode)
//
.ifdef RVY
c.ly a0, 16(a1)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.ly	a0, 16(a1)
// CHECK-CAP-32-SAME: # encoding: [0x88,0x69]
// CHECK-CAP-64-SAME: # encoding: [0x88,0x29]
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_LY_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_LY_RV64{{$}}
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-CAP-NEXT: #  <MCOperand Imm:16>>
c.sy a0, 16(a1)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.sy	a0, 16(a1)
// CHECK-CAP-32-SAME: # encoding: [0x88,0xe9]
// CHECK-CAP-64-SAME: # encoding: [0x88,0xa9]
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_SY_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_SY_RV64{{$}}
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-CAP-NEXT: #  <MCOperand Imm:16>>
c.lysp a0, 16(sp)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.lysp a0, 16(sp)
// CHECK-CAP-32-SAME: # encoding: [0x42,0x65]
// CHECK-CAP-64-SAME: # encoding: [0x42,0x25]
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_LYSP_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_LYSP_RV64{{$}}
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-CAP-NEXT: #  <MCOperand Imm:16>>
c.sysp a0, 16(sp)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.sysp a0, 16(sp)
// CHECK-CAP-32-SAME: # encoding: [0x2a,0xe8]
// CHECK-CAP-64-SAME: # encoding: [0x2a,0xa8]
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_SYSP_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_SYSP_RV64{{$}}
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-CAP-NEXT: #  <MCOperand Imm:16>>
.endif

///
/// Check that the compress patterns work as expected:
///

lb a0, 16(a1)
/// There is no c.lbu, so the first one should not be compressed:
// CHECK-ASM-AND-OBJ: lb	a0, 16(a1)
// CHECK-ASM-SAME: # encoding: [0x03,0x85,0x05,0x01]
// CHECK-ASM-NEXT: # <MCInst #[[#]] LB{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
lbu a0, 1(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lbu	a0, 1(a1)
// CHECK-ASM-SAME: # encoding: [0xc8,0x81]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LBU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
lh a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lh	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xe8,0x85]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LH{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
lhu a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lhu	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xa8,0x85]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LHU{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
lw a0, 16(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.lw	a0, 16(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0x49]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
sb a0, 1(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sb	a0, 1(a1)
// CHECK-ASM-SAME: # encoding: [0xc8,0x89]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SB{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
sh a0, 2(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sh	a0, 2(a1)
// CHECK-ASM-SAME: # encoding: [0xa8,0x8d]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SH{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:2>>
sw a0, 16(a1)
// CHECK-ASM-AND-OBJ-NEXT: c.sw	a0, 16(a1)
// CHECK-ASM-SAME: # encoding: [0x88,0xc9]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
//
/// Capability load & store
//
ly a0, 16(a1)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.ly	a0, 16(a1)
// CHECK-INT-ASM-AND-OBJ-NEXT: ly	a0, 16(a1)
// CHECK-INT-SAME: # encoding: [0x7b,0x95,0x05,0x01]
// CHECK-CAP-32-SAME: # encoding: [0x88,0x69]
// CHECK-CAP-64-SAME: # encoding: [0x88,0x29]
// CHECK-INT-NEXT: # <MCInst #[[#]] LY{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_LY_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_LY_RV64{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
sy a0, 16(a1)
// CHECK-INT-ASM-AND-OBJ-NEXT: sy	a0, 16(a1)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.sy	a0, 16(a1)
// CHECK-INT-SAME: # encoding: [0x7b,0xa8,0xa5,0x00]
// CHECK-CAP-32-SAME: # encoding: [0x88,0xe9]
// CHECK-CAP-64-SAME: # encoding: [0x88,0xa9]
// CHECK-INT-NEXT: # <MCInst #[[#]] SY{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_SY_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_SY_RV64{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X11>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X11_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
//
/// Test c.l*sp/c.s*sp compress patters (the *y ones only compress in RVY mode):
//
lw a0, 16(sp)
// CHECK-ASM-AND-OBJ-NEXT: c.lwsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x42,0x45]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_LWSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
sw a0, 16(sp)
// CHECK-ASM-AND-OBJ-NEXT: c.swsp	a0, 16(sp)
// CHECK-ASM-SAME: # encoding: [0x2a,0xc8]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_SWSP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
ly a0, 16(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: ly	a0, 16(sp)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.lysp	a0, 16(sp)
// CHECK-INT-SAME: # encoding: [0x7b,0x15,0x01,0x01]
// CHECK-CAP-32-SAME: # encoding: [0x42,0x65]
// CHECK-CAP-64-SAME: # encoding: [0x42,0x25]
// CHECK-INT-NEXT: # <MCInst #[[#]] LY{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_LYSP_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_LYSP_RV64{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
sy a0, 16(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: sy	a0, 16(sp)
// CHECK-CAP-ASM-AND-OBJ-NEXT: c.sysp	a0, 16(sp)
// CHECK-INT-SAME: # encoding:  [0x7b,0x28,0xa1,0x00]
// CHECK-CAP-32-SAME: # encoding: [0x2a,0xe8]
// CHECK-CAP-64-SAME: # encoding: [0x2a,0xa8]
// CHECK-INT-NEXT: # <MCInst #[[#]] SY{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_SYSP_RV32{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] C_SYSP_RV64{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>

// TODO: Test the pseudo expansions using AUIPC:
// lb a0, sym
// lbu a0, sym
// lh a0, sym
// lhu a0, sym
// lw a0, sym
// ly a0, sym
//
// sb a0, sym, t0
// sh a0, sym, t0
// sw a0, sym, t0
// sy a0, sym, t0
