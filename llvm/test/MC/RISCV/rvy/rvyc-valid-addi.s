/// Check that addi/yaddi uses compressed instructions correctly depending on the mode.
// RUN: llvm-mc --triple=riscv32 --mattr=+zca,+zcb,+zcf,+zcd,+f,+d,+zfh,+xllvmrvyipm --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv32 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+zca,+zcb,+zcf,+zcd,+f,+d,+zfh,+xllvmrvyipm --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump --mattr=+xllvmrvyipm -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-CAP-ASM-AND-OBJ
//
// RUN: llvm-mc --triple=riscv64 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+xllvmrvyipm --defsym=RV64=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-INT,CHECK-INT-ASM-AND-OBJ %s
// RUN: llvm-mc --triple=riscv64 --mattr=+zca,+zcb,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --defsym=RV64=1 --defsym=RV64_CAP=1 --riscv-no-aliases --show-encoding --show-inst < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-ASM,CHECK-CAP,CHECK-CAP-ASM-AND-OBJ %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+zca,+zcb,+zcd,+f,+d,+zfh,+xllvmrvyipm --defsym=RV64=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump --mattr=+xllvmrvyipm -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-INT-ASM-AND-OBJ
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+zca,+zcb,+f,+d,+zfh,+experimental-y --defsym=RVY=1 --defsym=RV64=1 --defsym=RV64_CAP=1 --riscv-add-build-attributes < %s \
// RUN:   | llvm-objdump -M no-aliases -d --no-print-imm-hex - | FileCheck %s --check-prefixes=CHECK-ASM-AND-OBJ,CHECK-CAP-ASM-AND-OBJ

c.addi4spn a0, sp, 12
// CHECK-ASM-AND-OBJ: c.addi4spn	a0, sp, 12
// CHECK-ASM-SAME: # encoding: [0x68,0x00]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_ADDI4SPN{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:X10>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:12>>

c.addi16sp sp, 16
// CHECK-ASM-AND-OBJ: c.addi16sp	sp, 16
// CHECK-ASM-SAME: # encoding: [0x41,0x61]
// CHECK-ASM-NEXT: # <MCInst #[[#]] C_ADDI16SP{{$}}
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-INT-NEXT: #  <MCOperand Reg:X2>
// CHECK-CAP-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>

addi a0, sp, 12
// CHECK-INT-ASM-AND-OBJ: c.addi4spn	a0, sp, 12
// CHECK-INT-SAME: # encoding: [0x68,0x00]
// CHECK-CAP-ASM-AND-OBJ: addi	a0, sp, 12
// CHECK-CAP-SAME: # encoding: [0x13,0x05,0xc1,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] C_ADDI4SPN{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] ADDI{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
// CHECK-ASM-NEXT: #  <MCOperand Reg:X2>
// CHECK-ASM-NEXT: #  <MCOperand Imm:12>>

yaddi a0, sp, 12
// CHECK-INT-ASM-AND-OBJ: yaddi	a0, sp, 12
// CHECK-INT-SAME: # encoding: [0x7b,0x45,0xc1,0x00]
// CHECK-CAP-ASM-AND-OBJ: c.addi4spn	a0, sp, 12
// CHECK-CAP-SAME: # encoding: [0x68,0x00]
// CHECK-INT-NEXT: # <MCInst #[[#]] YADDI{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] C_ADDI4SPN{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
// CHECK-ASM-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:12>>

yaddi sp, sp, 16
// CHECK-INT-ASM-AND-OBJ: yaddi	sp, sp, 16
// CHECK-INT-SAME: # encoding: [0x7b,0x41,0x01,0x01]
// CHECK-CAP-ASM-AND-OBJ: c.addi16sp	sp, 16
// CHECK-CAP-SAME: # encoding: [0x41,0x61]
// CHECK-INT-NEXT: # <MCInst #[[#]] YADDI{{$}}
// CHECK-CAP-NEXT: # <MCInst #[[#]] C_ADDI16SP{{$}}
// CHECK-ASM-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Reg:X2_Y>
// CHECK-ASM-NEXT: #  <MCOperand Imm:16>>
