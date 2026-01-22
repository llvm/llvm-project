# RUN: llvm-mc --triple=riscv32 -mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-ASM-32 -D"#XLEN=32" %s
# RUN: llvm-mc --triple=riscv32 -mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-ASM-32 -D"#XLEN=32" %s
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex - | FileCheck %s -D"#XLEN=32"
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y,+cap-mode < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex - | FileCheck %s -D"#XLEN=32"

# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --show-inst --defsym=RV64=1 < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-ASM-64 -D"#XLEN=64" %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst --defsym=RV64=1 < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-ASM-64 -D"#XLEN=64" %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y --defsym=RV64=1 < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex - | FileCheck %s -D"#XLEN=64"
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y,+cap-mode --defsym=RV64=1 < %s \
# RUN:   | llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex - | FileCheck %s -D"#XLEN=64"

# CHECK: addy		a0, a0, a1
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0xb5,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] ADDY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X11>>
addy a0, a0, a1
# CHECK-NEXT: addiy		a0, a0, 12
# CHECK-ASM-SAME: # encoding: [0x1b,0x25,0xc5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] ADDIY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:12>>
addiy a0, a0, 12
# CHECK-NEXT: addiy		a0, a0, 12
# CHECK-ASM-SAME: # encoding: [0x1b,0x25,0xc5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] ADDIY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:12>>
addy a0, a0, 12
# CHECK-NEXT: yaddrw		a0, a0, a1
# CHECK-ASM-SAME: # encoding: [0x33,0x15,0xb5,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YADDRW{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X11>>
yaddrw a0, a0, a1
# CHECK-NEXT: ypermc		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x25,0xa5,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YPERMC{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
ypermc a0, a0, a0
# CHECK-NEXT: ymv		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x05,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YMV{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ymv a0, a0
## Note: mv expands to integer addi and not capability ymv:
# CHECK-NEXT: addi		a0, a0, 0
# CHECK-ASM-SAME: # encoding: [0x13,0x05,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] ADDI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
mv a0, a0
# CHECK-NEXT: packy		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x35,0xa5,0x08]
# CHECK-ASM-NEXT: # <MCInst #[[#]] PACKY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
packy a0, a0, a0
# CHECK-NEXT: packy		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x35,0xa5,0x08]
# CHECK-ASM-NEXT: # <MCInst #[[#]] PACKY{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
yhiw a0, a0, a0
# CHECK-NEXT: ybndsw		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0xa5,0x0e]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSW{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
ybndsw a0, a0, a0
# CHECK-NEXT: ybndsrw		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x15,0xa5,0x0e]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSRW{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
ybndsrw a0, a0, a0
# CHECK-NEXT: ybndswi		a0, a0, 12
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xb5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:12>>
ybndswi a0, a0, 12
# CHECK-NEXT: ybndswi		a0, a0, 12
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xb5,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:12>>
ybndswi a0, a0, 12
## Test all the  min and max values for the ybndswi encoding
# CHECK-NEXT: ybndswi		a0, a0, 1
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0x05,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
ybndswi a0, a0, 1
# CHECK-NEXT: ybndswi		a0, a0, 256
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xf5,0x0f]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:256>>
ybndswi a0, a0, 256
# CHECK-NEXT: ybndswi		a0, a0, 258
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0x05,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:258>>
ybndswi a0, a0, 258
# CHECK-NEXT: ybndswi		a0, a0, 768
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xf5,0x1f]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:768>>
ybndswi a0, a0, 768
# CHECK-NEXT: ybndswi		a0, a0, 772
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0x05,0x20]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:772>>
ybndswi a0, a0, 772
# CHECK-NEXT: ybndswi		a0, a0, 1792
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xf5,0x2f]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:1792>>
ybndswi a0, a0, 1792
# CHECK-NEXT: ybndswi		a0, a0, 1800
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0x05,0x30]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:1800>>
ybndswi a0, a0, 1800
# CHECK-NEXT: ybndswi		a0, a0, 3840
# CHECK-ASM-SAME: # encoding: [0x1b,0x35,0xf5,0x3f]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBNDSWI{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:3840>>
ybndswi a0, a0, 3840
# CHECK-NEXT: ysunseal		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x25,0xa5,0x0e]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YSUNSEAL{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ysunseal a0, a0, a0
# CHECK-NEXT: ybaser		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x55,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YBASER{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ybaser a0, a0
# CHECK-NEXT: ylenr		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x65,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YLENR{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ylenr a0, a0
# CHECK-NEXT: ytagr		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x05,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YTAGR{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ytagr a0, a0
# CHECK-NEXT: ypermr		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x15,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YPERMR{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ypermr a0, a0
# CHECK-NEXT: ytyper		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x25,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YTYPER{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ytyper a0, a0
# CHECK-NEXT: srliy		a0, a0, [[#XLEN]]
# CHECK-ASM-64-SAME: # encoding: [0x13,0x55,0x05,0x04]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SRLIY[[#XLEN]]{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:[[#XLEN]]>>
.ifdef RV64
srliy a0, a0, 64
.else
srliy a0, a0, 32
.endif
# CHECK-NEXT: srliy		a0, a0, [[#XLEN]]
# CHECK-ASM-32-SAME: # encoding: [0x13,0x55,0x05,0x02]
# CHECK-ASM-64-SAME: # encoding: [0x13,0x55,0x05,0x04]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SRLIY[[#XLEN]]{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:[[#XLEN]]>>
yhir a0, a0
# CHECK-NEXT: syeq		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x45,0xa5,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] SYEQ{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
syeq a0, a0, a0
# CHECK-NEXT: ylt		a0, a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x65,0xa5,0x0c]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YLT{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10_Y>>
ylt a0, a0, a0
# CHECK-NEXT: yamask		a0, a0
# CHECK-ASM-SAME: # encoding: [0x33,0x05,0x75,0x10]
# CHECK-ASM-NEXT: # <MCInst #[[#]] YAMASK{{$}}
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>
# CHECK-ASM-NEXT: #  <MCOperand Reg:X10>>
yamask a0, a0
