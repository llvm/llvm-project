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

# CHECK-INST:  ball    a1, a3, LBL0
# CHECK: encoding: [0x37,0x41,A]
 ball a1, a3, LBL0
# CHECK-INST:  bany    a8, a13, LBL0
# CHECK: encoding: [0xd7,0x88,A]
 bany a8, a13, LBL0
# CHECK-INST:  bbc     a8, a7, LBL0
# CHECK: encoding: [0x77,0x58,A]
 bbc a8, a7, LBL0
# CHECK-INST:  bbci    a3, 16, LBL0
# CHECK: encoding: [0x07,0x73,A]
 bbci a3, 16, LBL0
# CHECK-INST:  bbs     a12, a5, LBL0
# CHECK: encoding: [0x57,0xdc,A]
 bbs a12, a5, LBL0
# CHECK-INST:  bbsi    a3, 16, LBL0
# CHECK: encoding: [0x07,0xf3,A]
 bbsi a3, 16, LBL0
# CHECK-INST:  bnall   a7, a3, LBL0
# CHECK: encoding: [0x37,0xc7,A]
 bnall a7, a3, LBL0
# CHECK-INST:  bnone   a2, a4, LBL0
# CHECK: encoding: [0x47,0x02,A]
 bnone a2, a4, LBL0
 
# CHECK-INST:  beq     a1, a2, LBL0
# CHECK: encoding: [0x27,0x11,A]
 beq a1, a2, LBL0
# CHECK-INST:  beq     a11, a5, LBL0
# CHECK: encoding: [0x57,0x1b,A]
 beq a11, a5, LBL0
# CHECK-INST:  beqi    a1, 256, LBL0 
# CHECK: encoding: [0x26,0xf1,A]
 beqi a1, 256, LBL0
# CHECK-INST:  beqi    a11, -1, LBL0 
# CHECK: encoding: [0x26,0x0b,A]
 beqi a11, -1, LBL0
# CHECK-INST:  beqz    a8, LBL0  
# CHECK: encoding: [0x16,0bAAAA1000,A]
 beqz a8, LBL0
# CHECK-INST:  bge     a14, a2, LBL0  
# CHECK: encoding: [0x27,0xae,A]
 bge a14, a2, LBL0
# CHECK-INST:  bgei    a11, -1, LBL0  
# CHECK: encoding: [0xe6,0x0b,A]
 bgei a11, -1, LBL0
# CHECK-INST:  bgei    a11, 128, LBL0  
# CHECK: encoding: [0xe6,0xeb,A]
 bgei a11, 128, LBL0
# CHECK-INST:  bgeu    a14, a2, LBL0  
# CHECK: encoding: [0x27,0xbe,A]
 bgeu a14, a2, LBL0
# CHECK-INST:  bgeui   a9, 32768, LBL0   
# CHECK: encoding: [0xf6,0x09,A]
 bgeui a9, 32768, LBL0
# CHECK-INST:  bgeui   a7, 65536, LBL0   
# CHECK: encoding: [0xf6,0x17,A]
 bgeui a7, 65536, LBL0
# CHECK-INST:  bgeui   a7, 64, LBL0   
# CHECK: encoding: [0xf6,0xd7,A]
 bgeui a7, 64, LBL0
# CHECK-INST:  bgez    a8, LBL0   
# CHECK: encoding: [0xd6,0bAAAA1000,A]
 bgez a8, LBL0
# CHECK-INST:  blt     a14, a2, LBL0  
# CHECK: encoding: [0x27,0x2e,A]
 blt a14, a2, LBL0
# CHECK-INST:  blti    a12, -1, LBL0  
# CHECK: encoding: [0xa6,0x0c,A]
 blti a12, -1, LBL0
# CHECK-INST:  blti    a0, 32, LBL0   
# CHECK: encoding: [0xa6,0xc0,A]
 blti a0, 32, LBL0
# CHECK-INST:  bgeu    a13, a1, LBL0   
# CHECK: encoding: [0x17,0xbd,A]
 bgeu a13, a1, LBL0
# CHECK-INST:  bltui   a7, 16, LBL0  
# CHECK: encoding: [0xb6,0xb7,A]
 bltui a7, 16, LBL0
# CHECK-INST:  bltz    a6, LBL0  
# CHECK: encoding: [0x96,0bAAAA0110,A]
 bltz a6, LBL0
# CHECK-INST:  bne     a3, a4, LBL0 
# CHECK: encoding: [0x47,0x93,A]
 bne a3, a4, LBL0
# CHECK-INST:  bnei    a5, 12, LBL0 
# CHECK: encoding: [0x66,0xa5,A]
 bnei a5, 12, LBL0
# CHECK-INST:  bnez    a5, LBL0 
# CHECK: encoding: [0x56,0bAAAA0101,A]
 bnez a5, LBL0	
	
# CHECK-INST:  call0   LBL0 
# CHECK: encoding: [0bAA000101,A,A]
 call0  LBL0
# CHECK-INST:  callx0  a1  
# CHECK: encoding: [0xc0,0x01,0x00]
 callx0 a1
# CHECK-INST:  dsync  
# CHECK: encoding: [0x30,0x20,0x00]
 dsync
# CHECK-INST:  esync  
# CHECK: encoding: [0x20,0x20,0x00]
 esync
 
# CHECK-INST:  extui    a1, a2, 7, 8 
# CHECK: encoding: [0x20,0x17,0x74]
 extui a1, a2, 7, 8

# CHECK-INST:  extw  
# CHECK: encoding: [0xd0,0x20,0x00]
 extw
  
# CHECK-INST:  isync 
# CHECK: encoding: [0x00,0x20,0x00]
 isync
  
# CHECK-INST:  j       LBL0 
# CHECK: encoding: [0bAA000110,A,A]
 j LBL0
# CHECK-INST:  jx      a2 
# CHECK: encoding: [0xa0,0x02,0x00]
 jx a2

# CHECK-INST:  l8ui    a2, a1, 3
# CHECK: encoding: [0x22,0x01,0x03]
 l8ui a2, sp, 3
# CHECK-INST:  l16si   a3, a1, 4
# CHECK: encoding: [0x32,0x91,0x02]
 l16si a3, sp, 4
# CHECK-INST: l16ui   a4, a1, 6
# CHECK: encoding: [0x42,0x11,0x03]
 l16ui a4, sp, 6
# CHECK-INST: l32i    a5, a1, 8
# CHECK: encoding: [0x52,0x21,0x02]
 l32i a5, sp, 8
# CHECK-INST: l32r    a6, LBL0 
# CHECK: encoding: [0x61,A,A]
 l32r a6, LBL0
 
# CHECK-INST: memw 
# CHECK: encoding: [0xc0,0x20,0x00]
 memw
 
# CHECK-INST: moveqz  a2, a3, a4 
# CHECK: encoding: [0x40,0x23,0x83]
 moveqz  a2,a3,a4
# CHECK-INST: movgez  a3, a11, a12 
# CHECK: encoding: [0xc0,0x3b,0xb3]
 movgez  a3,a11,a12

# CHECK-INST: movi    a1, -2048 
# CHECK: encoding: [0x12,0xa8,0x00]
 movi  a1, -2048

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
  
# CHECK-INST: ret
# CHECK: encoding: [0x80,0x00,0x00]
 ret
  
# CHECK-INST: rsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
 rsr a8, sar
# CHECK-INST: rsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
 rsr.sar a8
 # CHECK-INST: rsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
 rsr a8, 3


# CHECK-INST: rsync 
# CHECK: encoding: [0x10,0x20,0x00] 
 rsync

# CHECK-INST: s8i     a2, a1, 3 
# CHECK: encoding: [0x22,0x41,0x03] 
 s8i a2, sp, 3
# CHECK-INST: s16i    a3, a1, 4 
# CHECK: encoding: [0x32,0x51,0x02] 
 s16i a3, sp, 4
# CHECK-INST: s32i    a5, a1, 8 
# CHECK: encoding: [0x52,0x61,0x02] 
 s32i a5, sp, 8

# CHECK-INST: sll     a10, a11 
# CHECK: encoding: [0x00,0xab,0xa1] 
 sll a10, a11
 
# CHECK-INST: slli    a5, a1, 15
# CHECK: encoding: [0x10,0x51,0x11]
 slli a5, a1, 15

# CHECK-INST: sra     a12, a3
# CHECK: encoding: [0x30,0xc0,0xb1]
 sra a12, a3

# CHECK-INST: srai    a8, a5, 0
# CHECK: encoding: [0x50,0x80,0x21]
 srai a8, a5, 0
 
# CHECK-INST: src     a3, a4, a5 
# CHECK: encoding: [0x50,0x34,0x81] 
 src a3, a4, a5

# CHECK-INST: srl     a6, a7 
# CHECK: encoding: [0x70,0x60,0x91]
 srl a6, a7

# CHECK-INST: srli    a3, a4, 8
# CHECK: encoding: [0x40,0x38,0x41]
 srli a3, a4, 8

# CHECK-INST: ssa8l   a14 
# CHECK: encoding: [0x00,0x2e,0x40] 
 ssa8l a14
 
# CHECK-INST: ssai    31
# CHECK: encoding: [0x10,0x4f,0x40]
 ssai 31

# CHECK-INST: ssl     a0
# CHECK: encoding: [0x00,0x10,0x40]
 ssl a0

# CHECK-INST: ssr     a2
# CHECK: encoding: [0x00,0x02,0x40]
 ssr a2
 
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

# CHECK-INST: wsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x13] 
 wsr a8, sar
# CHECK-INST: wsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x13] 
 wsr.sar a8
# CHECK-INST: wsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x13]  
 wsr a8, 3

# CHECK-INST: xor     a6, a4, a5
# CHECK: encoding: [0x50,0x64,0x30] 
 xor a6, a4, a5

# CHECK-INST: xsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x61] 
 xsr a8, sar
# CHECK-INST: xsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x61
 xsr.sar a8
# CHECK-INST: xsr     a8, sar
# CHECK: encoding: [0x80,0x03,0x61
 xsr a8, 3
