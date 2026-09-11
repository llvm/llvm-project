// RUN: llvm-mc --triple=riscv32 --mattr=+zca,+zcb,+zcf,+zcd,+f,+d,+zfh,+xllvmrvyipm --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-32,CHECK-INT-ASM-AND-OBJ,CHECK-INT-32-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv32 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-32,CHECK-CAP-ASM-AND-OBJ,CHECK-CAP-32-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+zca,+zcb,+zcf,+zcd,+f,+d,+zfh,+xllvmrvyipm --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump --mattr=+xllvmrvyipm -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-32-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-CAP-32-ASM-AND-OBJ,CHECK-CAP-ASM-AND-OBJ

// RUN: llvm-mc --triple=riscv64 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+xllvmrvyipm --defsym=RV64=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-64,CHECK-INT-64-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv64 --mattr=+zca,+zcb,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --defsym=RV64=1 --defsym=RV64_CAP=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-64,CHECK-CAP-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+xllvmrvyipm --defsym=RV64=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump --mattr=+xllvmrvyipm -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-64-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+zca,+zcb,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --defsym=RV64=1 --defsym=RV64_CAP=1 --riscv-add-build-attributes < %s \
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
/// Compressed Floating Point load & store (RV32 and RV64)
//
.ifndef RV64
.ifndef RVY
c.flw fa5, 0(a0)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.flw	fa5, 0(a0)
// CHECK-INT-32-SAME: # encoding: [0x1c,0x61]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FLW{{$}}
// CHECK-INT-32-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-32-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-32-NEXT: #  <MCOperand Imm:0>>
c.fsw fa5, 0(a0)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.fsw	fa5, 0(a0)
// CHECK-INT-32-SAME: # encoding: [0x1c,0xe1]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FSW{{$}}
// CHECK-INT-32-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-32-NEXT: #  <MCOperand Reg:X10>
// CHECK-INT-32-NEXT: #  <MCOperand Imm:0>>
c.flwsp fa5, 0(sp)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.flwsp	fa5, 0(sp)
// CHECK-INT-32-SAME: # encoding: [0x82,0x67]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FLWSP{{$}}
// CHECK-INT-32-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-32-NEXT: #  <MCOperand Reg:X2>
// CHECK-INT-32-NEXT: #  <MCOperand Imm:0>>
c.fswsp fa5, 0(sp)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.fswsp	fa5, 0(sp)
// CHECK-INT-32-SAME: # encoding: [0x3e,0xe0]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FSWSP{{$}}
// CHECK-INT-32-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-32-NEXT: #  <MCOperand Reg:X2>
// CHECK-INT-32-NEXT: #  <MCOperand Imm:0>>
.endif
.endif
//
.ifndef RV64_CAP
c.fld fa5, 0(a0)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fld	fa5, 0(a0)
// CHECK-CAP-32-ASM-AND-OBJ-NEXT: c.fld	fa5, 0(a0)
// CHECK-INT-SAME: # encoding: [0x1c,0x21]
// CHECK-CAP-32-SAME: # encoding: [0x1c,0x21]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FLD{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FLD{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Imm:0>>
// CHECK-CAP-32-NEXT: #  <MCOperand Imm:0>>
c.fsd fa5, 0(a0)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fsd	fa5, 0(a0)
// CHECK-CAP-32-ASM-AND-OBJ-NEXT: c.fsd	fa5, 0(a0)
// CHECK-INT-SAME: # encoding: [0x1c,0xa1]
// CHECK-CAP-32-SAME: # encoding: [0x1c,0xa1]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FSD{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FSD{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Imm:0>>
// CHECK-CAP-32-NEXT: #  <MCOperand Imm:0>>
c.fldsp fa5, 0(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fldsp	fa5, 0(sp)
// CHECK-CAP-32-ASM-AND-OBJ-NEXT: c.fldsp	fa5, 0(sp)
// CHECK-INT-SAME: # encoding: [0x82,0x27]
// CHECK-CAP-32-SAME: # encoding: [0x82,0x27]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FLDSP{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FLDSP{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-INT-NEXT: #  <MCOperand Imm:0>>
// CHECK-CAP-32-NEXT: #  <MCOperand Imm:0>>
c.fsdsp fa5, 0(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fsdsp	fa5, 0(sp)
// CHECK-CAP-32-ASM-AND-OBJ-NEXT: c.fsdsp	fa5, 0(sp)
// CHECK-INT-SAME: # encoding: [0x3e,0xa0]
// CHECK-CAP-32-SAME: # encoding: [0x3e,0xa0]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FSDSP{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FSDSP{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-32-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-INT-NEXT: #  <MCOperand Imm:0>>
// CHECK-CAP-32-NEXT: #  <MCOperand Imm:0>>
.endif
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
//
/// Memory-relative float compress patterns
//
flw fa5, 0(a0)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.flw	fa5, 0(a0)
// CHECK-CAP-ASM-AND-OBJ-NEXT: flw	fa5, 0(a0)
// CHECK-INT-64-ASM-AND-OBJ-NEXT: flw	fa5, 0(a0)
// CHECK-INT-32-SAME: # encoding: [0x1c,0x61]
// CHECK-CAP-SAME: # encoding: [0x87,0x27,0x05,0x00]
// CHECK-INT-64-SAME: # encoding: [0x87,0x27,0x05,0x00]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FLW{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] FLW{{$}}
// CHECK-INT-64-NEXT: # <MCInst #[[#]] FLW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fsw fa5, 0(a0)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.fsw	fa5, 0(a0)
// CHECK-CAP-ASM-AND-OBJ-NEXT: fsw	fa5, 0(a0)
// CHECK-INT-64-ASM-AND-OBJ-NEXT: fsw	fa5, 0(a0)
// CHECK-INT-32-SAME: # encoding: [0x1c,0xe1]
// CHECK-CAP-SAME: # encoding: [0x27,0x20,0xf5,0x00]
// CHECK-INT-64-SAME: # encoding: [0x27,0x20,0xf5,0x00]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FSW{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] FSW{{$}}
// CHECK-INT-64-NEXT: # <MCInst #[[#]] FSW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fld fa5, 0(a0)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fld	fa5, 0(a0)
// CHECK-CAP-32-NEXT: c.fld	fa5, 0(a0)
// CHECK-CAP-64-NEXT: fld	fa5, 0(a0)
// CHECK-INT-SAME: # encoding: [0x1c,0x21]
// CHECK-CAP-32-SAME: # encoding: [0x1c,0x21]
// CHECK-CAP-64-SAME: # encoding: [0x87,0x37,0x05,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FLD{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FLD{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] FLD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fsd fa5, 0(a0)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fsd	fa5, 0(a0)
// CHECK-CAP-32-NEXT: c.fsd	fa5, 0(a0)
// CHECK-CAP-64-NEXT: fsd	fa5, 0(a0)
// CHECK-INT-SAME: # encoding: [0x1c,0xa1]
// CHECK-CAP-32-SAME: # encoding: [0x1c,0xa1]
// CHECK-CAP-64-SAME: # encoding: [0x27,0x30,0xf5,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FSD{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FSD{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] FSD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
//
/// Stack-relative float compress patterns
//
flw fa5, 0(sp)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.flwsp	fa5, 0(sp)
// CHECK-CAP-ASM-AND-OBJ: flw	fa5, 0(sp)
// CHECK-INT-64-ASM-AND-OBJ: flw	fa5, 0(sp)
// CHECK-INT-32-SAME: # encoding: [0x82,0x67]
// CHECK-CAP-SAME: # encoding: [0x87,0x27,0x01,0x00]
// CHECK-INT-64-SAME: # encoding: [0x87,0x27,0x01,0x00]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FLWSP{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] FLW{{$}}
// CHECK-INT-64-NEXT: # <MCInst #[[#]] FLW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fsw fa5, 0(sp)
// CHECK-INT-32-ASM-AND-OBJ-NEXT: c.fswsp	fa5, 0(sp)
// CHECK-CAP-ASM-AND-OBJ-NEXT: fsw	fa5, 0(sp)
// CHECK-INT-64-ASM-AND-OBJ-NEXT: fsw	fa5, 0(sp)
// CHECK-INT-32-SAME: # encoding: [0x3e,0xe0]
// CHECK-CAP-SAME: # encoding: [0x27,0x20,0xf1,0x00]
// CHECK-INT-64-SAME: # encoding: [0x27,0x20,0xf1,0x00]
// CHECK-INT-32-NEXT: # <MCInst #[[#]] C_FSWSP{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] FSW{{$}}
// CHECK-INT-64-NEXT: # <MCInst #[[#]] FSW{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_F>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fld fa5, 0(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fldsp	fa5, 0(sp)
// CHECK-CAP-32-NEXT: c.fldsp	fa5, 0(sp)
// CHECK-CAP-64-NEXT: fld	fa5, 0(sp)
// CHECK-INT-SAME: # encoding: [0x82,0x27]
// CHECK-CAP-32-SAME: # encoding: [0x82,0x27]
// CHECK-CAP-64-SAME: # encoding: [0x87,0x37,0x01,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FLDSP{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FLDSP{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] FLD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
fsd fa5, 0(sp)
// CHECK-INT-ASM-AND-OBJ-NEXT: c.fsdsp	fa5, 0(sp)
// CHECK-CAP-32-NEXT: c.fsdsp	fa5, 0(sp)
// CHECK-CAP-64-NEXT: fsd	fa5, 0(sp)
// CHECK-INT-SAME: # encoding: [0x3e,0xa0]
// CHECK-CAP-32-SAME: # encoding: [0x3e,0xa0]
// CHECK-CAP-64-SAME: # encoding: [0x27,0x30,0xf1,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_FSDSP{{$}}
// CHECK-CAP-32-NEXT: # <MCInst #[[#]] C_FSDSP{{$}}
// CHECK-CAP-64-NEXT: # <MCInst #[[#]] FSD{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:F15_D>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
