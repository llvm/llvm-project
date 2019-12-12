# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# CHECK-INST:  abs     a5, a6
# CHECK: encoding: [0x60,0x51,0x60]
 abs a5, a6

# CHECK-INST:  add   a3, a9, a4
# CHECK: encoding: [0x40,0x39,0x80]
 add a3, a9, a4
# CHECK-INST:  add a15, a9, a1
# CHECK: encoding: [0x10,0xf9,0x80] 
 add a15, a9, sp
	
# CHECK-INST:  addi a8, a1, -128
# CHECK: encoding: [0x82,0xc1,0x80]
 addi a8, sp, -128
# CHECK-INST:  addi a8, a1,  -12
# CHECK: encoding: [0x82,0xc1,0xf4]
 addi a8, a1,  -12
	
# CHECK-INST:  addmi a1, a2, 32512 
# CHECK: encoding: [0x12,0xd2,0x7f]	
 addmi a1, a2, 32512 
	
# CHECK-INST:  addx2   a2, a1, a5
# CHECK: encoding: [0x50,0x21,0x90]
 addx2 a2, sp, a5
# CHECK-INST:  addx4   a3, a1, a6
# CHECK: encoding: [0x60,0x31,0xa0]
 addx4 a3, sp, a6
# CHECK-INST:  addx8   a4, a1, a7
# CHECK: encoding: [0x70,0x41,0xb0]
 addx8 a4, sp, a7

# CHECK-INST:  dsync  
# CHECK: encoding: [0x30,0x20,0x00]
 dsync
# CHECK-INST:  esync  
# CHECK: encoding: [0x20,0x20,0x00]
 esync

# CHECK-INST:  extw  
# CHECK: encoding: [0xd0,0x20,0x00]
 extw
  
# CHECK-INST:  isync 
# CHECK: encoding: [0x00,0x20,0x00]
 isync
 
# CHECK-INST: memw 
# CHECK: encoding: [0xc0,0x20,0x00]
 memw
 
# CHECK-INST: moveqz  a2, a3, a4 
# CHECK: encoding: [0x40,0x23,0x83]
 moveqz  a2,a3,a4
# CHECK-INST: movgez  a3, a11, a12 
# CHECK: encoding: [0xc0,0x3b,0xb3]
 movgez  a3,a11,a12

# CHECK-INST: movltz  a7, a8, a9 
# CHECK: encoding: [0x90,0x78,0xa3]
 movltz a7, a8, a9
# CHECK-INST: movnez  a10, a11, a12
# CHECK: encoding: [0xc0,0xab,0x93]
 movnez a10,a11, a12
  
# CHECK-INST: neg     a1, a3
# CHECK: encoding: [0x30,0x10,0x60]
 neg a1, a3

# CHECK-INST: nop
# CHECK: encoding: [0xf0,0x20,0x00]
 nop

# CHECK-INST: or      a4, a5, a6
# CHECK: encoding: [0x60,0x45,0x20]
 or a4, a5, a6

# CHECK-INST: rsync 
# CHECK: encoding: [0x10,0x20,0x00] 
 rsync

# CHECK-INST: sub     a8, a2, a1 
# CHECK: encoding: [0x10,0x82,0xc0] 
 sub  a8, a2, a1
# CHECK-INST: subx2   a2, a1, a5 
# CHECK: encoding: [0x50,0x21,0xd0]  
 subx2 a2, sp, a5
# CHECK-INST: subx4   a3, a1, a6 
# CHECK: encoding: [0x60,0x31,0xe0] 
 subx4 a3, sp, a6
# CHECK-INST: subx8   a4, a1, a7
# CHECK: encoding: [0x70,0x41,0xf0] 
 subx8 a4, sp, a7

# CHECK-INST: xor     a6, a4, a5
# CHECK: encoding: [0x50,0x64,0x30] 
 xor a6, a4, a5

