# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+highpriinterrupts,+highpriinterrupts-level7 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: rfi 1
# CHECK: encoding: [0x10,0x31,0x00]
rfi 1

#Instruction format RRR
#CHECK-INST: xsr a2, epc2
#CHECK: encoding: [0x20,0xb2,0x61]
xsr a2,epc2

#Instruction format RRR
#CHECK-INST: xsr a2, epc2
#CHECK: encoding: [0x20,0xb2,0x61]
xsr.epc2 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc2
#CHECK: encoding: [0x20,0xb2,0x61]
xsr a2, 178

#Instruction format RRR
#CHECK-INST: xsr a2, epc3
#CHECK: encoding: [0x20,0xb3,0x61]
xsr a2,epc3

#Instruction format RRR
#CHECK-INST: xsr a2, epc3
#CHECK: encoding: [0x20,0xb3,0x61]
xsr.epc3 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc3
#CHECK: encoding: [0x20,0xb3,0x61]
xsr a2, 179

#Instruction format RRR
#CHECK-INST: xsr a2, epc4
#CHECK: encoding: [0x20,0xb4,0x61]
xsr a2,epc4

#Instruction format RRR
#CHECK-INST: xsr a2, epc4
#CHECK: encoding: [0x20,0xb4,0x61]
xsr.epc4 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc4
#CHECK: encoding: [0x20,0xb4,0x61]
xsr a2, 180

#Instruction format RRR
#CHECK-INST: xsr a2, epc5
#CHECK: encoding: [0x20,0xb5,0x61]
xsr a2,epc5

#Instruction format RRR
#CHECK-INST: xsr a2, epc5
#CHECK: encoding: [0x20,0xb5,0x61]
xsr.epc5 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc5
#CHECK: encoding: [0x20,0xb5,0x61]
xsr a2, 181

#Instruction format RRR
#CHECK-INST: xsr a2, epc6
#CHECK: encoding: [0x20,0xb6,0x61]
xsr a2,epc6

#Instruction format RRR
#CHECK-INST: xsr a2, epc6
#CHECK: encoding: [0x20,0xb6,0x61]
xsr.epc6 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc6
#CHECK: encoding: [0x20,0xb6,0x61]
xsr a2, 182

#Instruction format RRR
#CHECK-INST: xsr a2, epc7
#CHECK: encoding: [0x20,0xb7,0x61]
xsr a2,epc7

#Instruction format RRR
#CHECK-INST: xsr a2, epc7
#CHECK: encoding: [0x20,0xb7,0x61]
xsr.epc7 a2

#Instruction format RRR
#CHECK-INST: xsr a2, epc7
#CHECK: encoding: [0x20,0xb7,0x61]
xsr a2, 183

#Instruction format RRR
#CHECK-INST: xsr a2, eps2
#CHECK: encoding: [0x20,0xc2,0x61]
xsr a2,eps2

#Instruction format RRR
#CHECK-INST: xsr a2, eps2
#CHECK: encoding: [0x20,0xc2,0x61]
xsr.eps2 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps2
#CHECK: encoding: [0x20,0xc2,0x61]
xsr a2, 194

#Instruction format RRR
#CHECK-INST: xsr a2, eps3
#CHECK: encoding: [0x20,0xc3,0x61]
xsr a2,eps3

#Instruction format RRR
#CHECK-INST: xsr a2, eps3
#CHECK: encoding: [0x20,0xc3,0x61]
xsr.eps3 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps3
#CHECK: encoding: [0x20,0xc3,0x61]
xsr a2, 195

#Instruction format RRR
#CHECK-INST: xsr a2, eps4
#CHECK: encoding: [0x20,0xc4,0x61]
xsr a2,eps4

#Instruction format RRR
#CHECK-INST: xsr a2, eps4
#CHECK: encoding: [0x20,0xc4,0x61]
xsr.eps4 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps4
#CHECK: encoding: [0x20,0xc4,0x61]
xsr a2, 196

#Instruction format RRR
#CHECK-INST: xsr a2, eps5
#CHECK: encoding: [0x20,0xc5,0x61]
xsr a2,eps5

#Instruction format RRR
#CHECK-INST: xsr a2, eps5
#CHECK: encoding: [0x20,0xc5,0x61]
xsr.eps5 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps5
#CHECK: encoding: [0x20,0xc5,0x61]
xsr a2, 197

#Instruction format RRR
#CHECK-INST: xsr a2, eps6
#CHECK: encoding: [0x20,0xc6,0x61]
xsr a2,eps6

#Instruction format RRR
#CHECK-INST: xsr a2, eps6
#CHECK: encoding: [0x20,0xc6,0x61]
xsr.eps6 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps6
#CHECK: encoding: [0x20,0xc6,0x61]
xsr a2, 198

#Instruction format RRR
#CHECK-INST: xsr a2, eps7
#CHECK: encoding: [0x20,0xc7,0x61]
xsr a2,eps7

#Instruction format RRR
#CHECK-INST: xsr a2, eps7
#CHECK: encoding: [0x20,0xc7,0x61]
xsr.eps7 a2

#Instruction format RRR
#CHECK-INST: xsr a2, eps7
#CHECK: encoding: [0x20,0xc7,0x61]
xsr a2, 199

#Instruction format RRR
#CHECK-INST: xsr a2, excsave2
#CHECK: encoding: [0x20,0xd2,0x61]
xsr a2,excsave2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave2
#CHECK: encoding: [0x20,0xd2,0x61]
xsr.excsave2 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave2
#CHECK: encoding: [0x20,0xd2,0x61]
xsr a2, 210

#Instruction format RRR
#CHECK-INST: xsr a2, excsave3
#CHECK: encoding: [0x20,0xd3,0x61]
xsr a2,excsave3

#Instruction format RRR
#CHECK-INST: xsr a2, excsave3
#CHECK: encoding: [0x20,0xd3,0x61]
xsr.excsave3 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave3
#CHECK: encoding: [0x20,0xd3,0x61]
xsr a2, 211

#Instruction format RRR
#CHECK-INST: xsr a2, excsave4
#CHECK: encoding: [0x20,0xd4,0x61]
xsr a2,excsave4

#Instruction format RRR
#CHECK-INST: xsr a2, excsave4
#CHECK: encoding: [0x20,0xd4,0x61]
xsr.excsave4 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave4
#CHECK: encoding: [0x20,0xd4,0x61]
xsr a2, 212

#Instruction format RRR
#CHECK-INST: xsr a2, excsave5
#CHECK: encoding: [0x20,0xd5,0x61]
xsr a2,excsave5

#Instruction format RRR
#CHECK-INST: xsr a2, excsave5
#CHECK: encoding: [0x20,0xd5,0x61]
xsr.excsave5 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave5
#CHECK: encoding: [0x20,0xd5,0x61]
xsr a2, 213

#Instruction format RRR
#CHECK-INST: xsr a2, excsave6
#CHECK: encoding: [0x20,0xd6,0x61]
xsr a2,excsave6

#Instruction format RRR
#CHECK-INST: xsr a2, excsave6
#CHECK: encoding: [0x20,0xd6,0x61]
xsr.excsave6 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave6
#CHECK: encoding: [0x20,0xd6,0x61]
xsr a2, 214

#Instruction format RRR
#CHECK-INST: xsr a2, excsave7
#CHECK: encoding: [0x20,0xd7,0x61]
xsr a2,excsave7

#Instruction format RRR
#CHECK-INST: xsr a2, excsave7
#CHECK: encoding: [0x20,0xd7,0x61]
xsr.excsave7 a2

#Instruction format RRR
#CHECK-INST: xsr a2, excsave7
#CHECK: encoding: [0x20,0xd7,0x61]
xsr a2, 215
