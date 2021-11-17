# RUN: llvm-mc  %s -triple=m88k-unknown-openbsd -show-encoding -mcpu=mc88100 | FileCheck %s

isns:
  # integer add
  add      %r0, %r1, %r2
  add.ci   %r1, %r2, %r3
  add.co   %r2, %r3, %r4
  add.cio  %r3, %r4, %r5
  add      %r4, %r5, 0
  add      %r4, %r5, 4096
# CHECK: add      %r0, %r1, %r2          | encoding: [0xf4,0x01,0x70,0x02]
# CHECK: add.ci   %r1, %r2, %r3          | encoding: [0xf4,0x22,0x72,0x03]
# CHECK: add.co   %r2, %r3, %r4          | encoding: [0xf4,0x43,0x71,0x04]
# CHECK: add.cio  %r3, %r4, %r5          | encoding: [0xf4,0x64,0x73,0x05]
# CHECK: add      %r4, %r5, 0            | encoding: [0x70,0x85,0x00,0x00]
# CHECK: add      %r4, %r5, 4096         | encoding: [0x70,0x85,0x10,0x00]

  # unsigned integer add
  addu     %r0, %r1, %r2
  addu.ci  %r1, %r2, %r3
  addu.co  %r2, %r3, %r4
  addu.cio %r3, %r4, %r5
  addu     %r4, %r5, 0
  addu     %r4, %r5, 4096
# CHECK: addu     %r0, %r1, %r2          | encoding: [0xf4,0x01,0x60,0x02]
# CHECK: addu.ci  %r1, %r2, %r3          | encoding: [0xf4,0x22,0x62,0x03]
# CHECK: addu.co  %r2, %r3, %r4          | encoding: [0xf4,0x43,0x61,0x04]
# CHECK: addu.cio %r3, %r4, %r5          | encoding: [0xf4,0x64,0x63,0x05]
# CHECK: addu     %r4, %r5, 0            | encoding: [0x60,0x85,0x00,0x00]
# CHECK: addu     %r4, %r5, 4096         | encoding: [0x60,0x85,0x10,0x00]

# logical and
  and      %r0, %r1, %r2
  and.c    %r1, %r2, %r3
  and      %r2, %r3, 0
  and      %r2, %r3, 4096
  and.u    %r2, %r3, 0
  and.u    %r2, %r3, 4096
# CHECK: and      %r0, %r1, %r2          | encoding: [0xf4,0x01,0x40,0x02]
# CHECK: and.c    %r1, %r2, %r3          | encoding: [0xf4,0x22,0x44,0x03]
# CHECK: and      %r2, %r3, 0            | encoding: [0x40,0x43,0x00,0x00]
# CHECK: and      %r2, %r3, 4096         | encoding: [0x40,0x43,0x10,0x00]
# CHECK: and.u    %r2, %r3, 0            | encoding: [0x44,0x43,0x00,0x00]
# CHECK: and.u    %r2, %r3, 4096         | encoding: [0x44,0x43,0x10,0x00]

  # branch on bit clear
  bb0      0, %r1, 0
  bb0      0, %r1, -10
  bb0      0, %r1, 10
  bb0      31, %r1, 0
  bb0      31, %r1, -10
  bb0      31, %r1, 10
  bb0.n    0, %r1, 0
# COM: CHECK: bb0      0, %r1, 0              | encoding: [0xd0,0x01,0x00,0x00]

  # branch on bit set
  bb1       0, %r1, 0
  bb1       0, %r1, -10
  bb1       0, %r1, 10
  bb1       31, %r1, 0
  bb1       31, %r1, -10
  bb1       31, %r1, 10
  bb1.n     0, %r1, 0

  # conditional branch
  bcnd       eq0, %r1, 0
  bcnd       eq0, %r1, 10
  bcnd       eq0, %r1, -10
  bcnd.n     eq0, %r1, 0
  bcnd.n     eq0, %r1, 10
  bcnd.n     eq0, %r1, -10
  bcnd       ne0, %r1, 0
  bcnd       ne0, %r1, 10
  bcnd       ne0, %r1, -10
  bcnd.n     ne0, %r1, 0
  bcnd.n     ne0, %r1, 10
  bcnd.n     ne0, %r1, -10
  bcnd       gt0, %r1, 0
  bcnd       gt0, %r1, 10
  bcnd       gt0, %r1, -10
  bcnd.n     gt0, %r1, 0
  bcnd.n     gt0, %r1, 10
  bcnd.n     gt0, %r1, -10
  bcnd       lt0, %r1, 0
  bcnd       lt0, %r1, 10
  bcnd       lt0, %r1, -10
  bcnd.n     lt0, %r1, 0
  bcnd.n     lt0, %r1, 10
  bcnd.n     lt0, %r1, -10
  bcnd       ge0, %r1, 0
  bcnd       ge0, %r1, 10
  bcnd       ge0, %r1, -10
  bcnd.n     ge0, %r1, 0
  bcnd.n     ge0, %r1, 10
  bcnd.n     ge0, %r1, -10
  bcnd       le0, %r1, 0
  bcnd       le0, %r1, 10
  bcnd       le0, %r1, -10
  bcnd.n     le0, %r1, 0
  bcnd.n     le0, %r1, 10
  bcnd.n     le0, %r1, -10
  # using m5 field
  bcnd       3, %r1, 0
  bcnd       3, %r1, 10
  bcnd       3, %r1, -10
  bcnd.n     3, %r1, 0
  bcnd.n     3, %r1, 10
  bcnd.n     3, %r1, -10

# uncoditional branch
  br       0
  br       -10
  br       10
  br.n     0
  br.n     -10
  br.n     10

# branch to subroutine
#  bsr       0
#  bsr       -10
#  bsr       10
#  bsr.n     0
#  bsr.n     -10
#  bsr.n     10

# clear bit field
  clr      %r1, %r2, 5<15>
  clr      %r1, %r2, %r3
  clr      %r1, %r2, 6
  clr      %r1, %r2, <6>
# CHECK: clr      %r1, %r2, 5<15>        | encoding: [0xf0,0x22,0x80,0xaf]
# CHECK: clr      %r1, %r2, %r3          | encoding: [0xf4,0x22,0x80,0x03]
# CHECK: clr      %r1, %r2, 0<6>         | encoding: [0xf0,0x22,0x80,0x06]
# CHECK: clr      %r1, %r2, 0<6>         | encoding: [0xf0,0x22,0x80,0x06]

# integer compare
  cmp      %r0, %r1, %r2
  cmp      %r0, %r2, 0
  cmp      %r0, %r2, 4096
# CHECK: cmp      %r0, %r1, %r2          | encoding: [0xf4,0x01,0x7c,0x02]
# CHECK: cmp      %r0, %r2, 0            | encoding: [0x7c,0x02,0x00,0x00]
# CHECK: cmp      %r0, %r2, 4096         | encoding: [0x7c,0x02,0x10,0x00]

# signed integer divide
  divs     %r0, %r1, %r2
  divs     %r0, %r1, 0
  divs     %r0, %r1, 4096
# CHECK: divs     %r0, %r1, %r2          | encoding: [0xf4,0x01,0x78,0x02]
# CHECK: divs     %r0, %r1, 0            | encoding: [0x78,0x01,0x00,0x00]
# CHECK: divs     %r0, %r1, 4096         | encoding: [0x78,0x01,0x10,0x00]

  # unsigned integer divide
  divu     %r0, %r1, %r2
  divu     %r0, %r1, 0
  divu     %r0, %r1, 10
# CHECK: divu     %r0, %r1, %r2          | encoding: [0xf4,0x01,0x68,0x02]
# CHECK: divu     %r0, %r1, 0            | encoding: [0x68,0x01,0x00,0x00]
# CHECK: divu     %r0, %r1, 10           | encoding: [0x68,0x01,0x00,0x0a]

# extract signed bit field
  ext      %r0, %r1, 10<5>
  ext      %r1, %r2, %r3
  ext      %r2, %r3, 6
  ext      %r2, %r3, <6>
# CHECK: ext      %r0, %r1, 10<5>        | encoding: [0xf0,0x01,0x91,0x45]
# CHECK: ext      %r1, %r2, %r3          | encoding: [0xf4,0x22,0x90,0x03]
# CHECK: ext      %r2, %r3, 0<6>         | encoding: [0xf0,0x43,0x90,0x06]
# CHECK: ext      %r2, %r3, 0<6>         | encoding: [0xf0,0x43,0x90,0x06]

# extract unsigned bit field
  extu     %r0, %r1, 10<5>
  extu     %r1, %r2, %r3
  extu     %r1, %r2, 6
  extu     %r1, %r2, <6>
# CHECK: extu     %r0, %r1, 10<5>        | encoding: [0xf0,0x01,0x99,0x45]
# CHECK: extu     %r1, %r2, %r3          | encoding: [0xf4,0x22,0x98,0x03]
# CHECK: extu     %r1, %r2, 0<6>         | encoding: [0xf0,0x22,0x98,0x06]
# CHECK: extu     %r1, %r2, 0<6>         | encoding: [0xf0,0x22,0x98,0x06]

  # floating point add
  fadd.sss %r0, %r1, %r2
  fadd.ssd %r0, %r1, %r2
  fadd.sds %r0, %r1, %r2
  fadd.sdd %r0, %r1, %r2
  fadd.dss %r0, %r1, %r2
  fadd.dsd %r0, %r1, %r2
  fadd.dds %r0, %r1, %r2
  fadd.ddd %r0, %r1, %r2
# CHECK: fadd.sss %r0, %r1, %r2          | encoding: [0x84,0x01,0x28,0x02]
# CHECK: fadd.ssd %r0, %r1, %r2          | encoding: [0x84,0x01,0x28,0x82]
# CHECK: fadd.sds %r0, %r1, %r2          | encoding: [0x84,0x01,0x2a,0x02]
# CHECK: fadd.sdd %r0, %r1, %r2          | encoding: [0x84,0x01,0x2a,0x82]
# CHECK: fadd.dss %r0, %r1, %r2          | encoding: [0x84,0x01,0x28,0x22]
# CHECK: fadd.dsd %r0, %r1, %r2          | encoding: [0x84,0x01,0x28,0xa2]
# CHECK: fadd.dds %r0, %r1, %r2          | encoding: [0x84,0x01,0x2a,0x22]
# CHECK: fadd.ddd %r0, %r1, %r2          | encoding: [0x84,0x01,0x2a,0xa2]

# floating point compare
#  fcmp.ss %r0, %r1, %r2
#  fcmp.sd %r0, %r1, %r2
#  fcmp.ds %r0, %r1, %r2
#  fcmp.dd %r0, %r1, %r2
# COM: CHECK: fcmp.ss %r0, %r1, %r2           | encoding: [0x84 01 38 02]
# COM: CHECK: fcmp.sd %r0, %r1, %r2           | encoding: [0x84 01 38 82]
# COM: CHECK: fcmp.ds %r0, %r1, %r2           | encoding: [0x84 01 3a 02]
# COM: CHECK: fcmp.dd %r0, %r1, %r2           | encoding: [0x84 01 3a 82]

# floating point divide
  fdiv.sss %r0, %r1, %r2
  fdiv.ssd %r0, %r1, %r2
  fdiv.sds %r0, %r1, %r2
  fdiv.sdd %r0, %r1, %r2
  fdiv.dss %r0, %r1, %r2
  fdiv.dsd %r0, %r1, %r2
  fdiv.dds %r0, %r1, %r2
  fdiv.ddd %r0, %r1, %r2
# CHECK: fdiv.sss %r0, %r1, %r2          | encoding: [0x84,0x01,0x70,0x02]
# CHECK: fdiv.ssd %r0, %r1, %r2          | encoding: [0x84,0x01,0x70,0x82]
# CHECK: fdiv.sds %r0, %r1, %r2          | encoding: [0x84,0x01,0x72,0x02]
# CHECK: fdiv.sdd %r0, %r1, %r2          | encoding: [0x84,0x01,0x72,0x82]
# CHECK: fdiv.dss %r0, %r1, %r2          | encoding: [0x84,0x01,0x70,0x22]
# CHECK: fdiv.dsd %r0, %r1, %r2          | encoding: [0x84,0x01,0x70,0xa2]
# CHECK: fdiv.dds %r0, %r1, %r2          | encoding: [0x84,0x01,0x72,0x22]
# CHECK: fdiv.ddd %r0, %r1, %r2          | encoding: [0x84,0x01,0x72,0xa2]

# find first bit clear
  ff0      %r1, %r7
# CHECK: ff0      %r1, %r7               | encoding: [0xf4,0x20,0xec,0x07]

# find first bit set
  ff1      %r3, %r8
# CHECK: ff1      %r3, %r8               | encoding: [0xf4,0x60,0xe8,0x08]

# load from floating-point control register
  fldcr    %r0, %fcr50
# CHECK: fldcr    %r0, %fcr50            | encoding: [0x80,0x00,0x4e,0x40]

  # convert integer to floating point
#  flt.ss   %r0, %r3
#  flt.ds   %r0, %r10
# COM: CHECK: flt.ss   %r0, %r3               | encoding: [0x84,0x00,0x20,0x03]
# COM: CHECK: flt.ds   %r0, %r10              | encoding: [0x84,0x00,0x20,0x2a]

  # floating point multiply
  fmul.sss %r0, %r1, %r2
  fmul.ssd %r0, %r1, %r2
  fmul.sds %r0, %r1, %r2
  fmul.sdd %r0, %r1, %r2
  fmul.dss %r0, %r1, %r2
  fmul.dsd %r0, %r1, %r2
  fmul.dds %r0, %r1, %r2
  fmul.ddd %r0, %r1, %r2
# CHECK: fmul.sss %r0, %r1, %r2          | encoding: [0x84,0x01,0x00,0x02]
# CHECK: fmul.ssd %r0, %r1, %r2          | encoding: [0x84,0x01,0x00,0x82]
# CHECK: fmul.sds %r0, %r1, %r2          | encoding: [0x84,0x01,0x02,0x02]
# CHECK: fmul.sdd %r0, %r1, %r2          | encoding: [0x84,0x01,0x02,0x82]
# CHECK: fmul.dss %r0, %r1, %r2          | encoding: [0x84,0x01,0x00,0x22]
# CHECK: fmul.dsd %r0, %r1, %r2          | encoding: [0x84,0x01,0x00,0xa2]
# CHECK: fmul.dds %r0, %r1, %r2          | encoding: [0x84,0x01,0x02,0x22]
# CHECK: fmul.ddd %r0, %r1, %r2          | encoding: [0x84,0x01,0x02,0xa2]

# store to floating point control register
  fstcr    %r0, %fcr50
# CHECK: fstcr    %r0, %fcr50            | encoding: [0x80,0x00,0x8e,0x40]

  # floating point subtract
  fsub.sss %r0, %r1, %r2
  fsub.ssd %r0, %r1, %r2
  fsub.sds %r0, %r1, %r2
  fsub.sdd %r0, %r1, %r2
  fsub.dss %r0, %r1, %r2
  fsub.dsd %r0, %r1, %r2
  fsub.dds %r0, %r1, %r2
  fsub.ddd %r0, %r1, %r2
# CHECK: fsub.sss %r0, %r1, %r2          | encoding: [0x84,0x01,0x30,0x02]
# CHECK: fsub.ssd %r0, %r1, %r2          | encoding: [0x84,0x01,0x30,0x82]
# CHECK: fsub.sds %r0, %r1, %r2          | encoding: [0x84,0x01,0x32,0x02]
# CHECK: fsub.sdd %r0, %r1, %r2          | encoding: [0x84,0x01,0x32,0x82]
# CHECK: fsub.dss %r0, %r1, %r2          | encoding: [0x84,0x01,0x30,0x22]
# CHECK: fsub.dsd %r0, %r1, %r2          | encoding: [0x84,0x01,0x30,0xa2]
# CHECK: fsub.dds %r0, %r1, %r2          | encoding: [0x84,0x01,0x32,0x22]
# CHECK: fsub.ddd %r0, %r1, %r2          | encoding: [0x84,0x01,0x32,0xa2]

# exchange floating point control register
  fxcr     %r0, %r1, %fcr50
# CHECK: fxcr     %r0, %r1, %fcr50       | encoding: [0x80,0x01,0xce,0x41]

  # illegal operation
  illop1
  illop2
  illop3
# CHECK: illop1                          | encoding: [0xf4,0x00,0xfc,0x01]
# CHECK: illop2                          | encoding: [0xf4,0x00,0xfc,0x02]
# CHECK: illop3                          | encoding: [0xf4,0x00,0xfc,0x03]

# round floating point to integer
  int.ss       %r0, %r1
  int.sd       %r10, %r2
# CHECK: int.ss       %r0, %r1           | encoding: [0x84,0x00,0x48,0x01]
# CHECK: int.sd       %r10, %r2          | encoding: [0x85,0x40,0x48,0x82]

  # unconditional jump
  jmp      %r0
  jmp.n    %r10
# CHECK: jmp      %r0                    | encoding: [0xf4,0x00,0xc0,0x00]
# CHECK: jmp.n    %r10                   | encoding: [0xf4,0x00,0xc4,0x0a]

# jump to subroutine
  jsr      %r10
  jsr.n    %r13
# CHECK: jsr      %r10                   | encoding: [0xf4,0x00,0xc8,0x0a]
# CHECK: jsr.n    %r13                   | encoding: [0xf4,0x00,0xcc,0x0d]

# load register from memory
  ld.b     %r0, %r1, 0
  ld.b     %r0, %r1, 4096
  ld.bu    %r0, %r1, 0
  ld.bu    %r0, %r1, 4096
  ld.h     %r0, %r1, 0
  ld.h     %r0, %r1, 4096
  ld.hu    %r0, %r1, 0
  ld.hu    %r0, %r1, 4096
  ld       %r0, %r1, 0
  ld       %r0, %r1, 4096
  ld.d     %r0, %r1, 0
  ld.d     %r0, %r1, 4096
# CHECK: ld.b     %r0, %r1, 0            | encoding: [0x1c,0x01,0x00,0x00]
# CHECK: ld.b     %r0, %r1, 4096         | encoding: [0x1c,0x01,0x10,0x00]
# CHECK: ld.bu    %r0, %r1, 0            | encoding: [0x0c,0x01,0x00,0x00]
# CHECK: ld.bu    %r0, %r1, 4096         | encoding: [0x0c,0x01,0x10,0x00]
# CHECK: ld.h     %r0, %r1, 0            | encoding: [0x18,0x01,0x00,0x00]
# CHECK: ld.h     %r0, %r1, 4096         | encoding: [0x18,0x01,0x10,0x00]
# CHECK: ld.hu    %r0, %r1, 0            | encoding: [0x08,0x01,0x00,0x00]
# CHECK: ld.hu    %r0, %r1, 4096         | encoding: [0x08,0x01,0x10,0x00]
# CHECK: ld       %r0, %r1, 0            | encoding: [0x14,0x01,0x00,0x00]
# CHECK: ld       %r0, %r1, 4096         | encoding: [0x14,0x01,0x10,0x00]
# CHECK: ld.d     %r0, %r1, 0            | encoding: [0x10,0x01,0x00,0x00]
# CHECK: ld.d     %r0, %r1, 4096         | encoding: [0x10,0x01,0x10,0x00]
  ld.b         %r0, %r1, %r2
  ld.bu        %r1, %r2, %r3
  ld.h         %r2, %r3, %r4
  ld.hu        %r3, %r4, %r5
  ld           %r4, %r5, %r6
  ld.d         %r5, %r6, %r7
  ld.b.usr     %r6, %r7, %r8
  ld.bu.usr    %r7, %r8, %r9
  ld.h.usr     %r8, %r9, %r1
  ld.hu.usr    %r9, %r1, %r2
  ld.usr       %r1, %r2, %r3
  ld.d.usr     %r2, %r3, %r4
# CHECK: ld.b         %r0, %r1, %r2      | encoding: [0xf4,0x01,0x1c,0x02]
# CHECK: ld.bu        %r1, %r2, %r3      | encoding: [0xf4,0x22,0x0c,0x03]
# CHECK: ld.h         %r2, %r3, %r4      | encoding: [0xf4,0x43,0x18,0x04]
# CHECK: ld.hu        %r3, %r4, %r5      | encoding: [0xf4,0x64,0x08,0x05]
# CHECK: ld           %r4, %r5, %r6      | encoding: [0xf4,0x85,0x14,0x06]
# CHECK: ld.d         %r5, %r6, %r7      | encoding: [0xf4,0xa6,0x10,0x07]
# CHECK: ld.b.usr     %r6, %r7, %r8      | encoding: [0xf4,0xc7,0x1d,0x08]
# CHECK: ld.bu.usr    %r7, %r8, %r9      | encoding: [0xf4,0xe8,0x0d,0x09]
# CHECK: ld.h.usr     %r8, %r9, %r1      | encoding: [0xf5,0x09,0x19,0x01]
# CHECK: ld.hu.usr    %r9, %r1, %r2      | encoding: [0xf5,0x21,0x09,0x02]
# CHECK: ld.usr       %r1, %r2, %r3      | encoding: [0xf4,0x22,0x15,0x03]
# CHECK: ld.d.usr     %r2, %r3, %r4      | encoding: [0xf4,0x43,0x11,0x04]

  ld.b         %r0, %r1[%r2]
  ld.bu        %r1, %r2[%r3]
  ld.h         %r2, %r3[%r4]
  ld.hu        %r3, %r4[%r5]
  ld           %r4, %r5[%r6]
  ld.d         %r5, %r6[%r7]
  ld.b.usr     %r6, %r7[%r8]
  ld.bu.usr    %r7, %r8[%r9]
  ld.h.usr     %r8, %r9[%r1]
  ld.hu.usr    %r9, %r1[%r2]
  ld.usr       %r1, %r2[%r3]
  ld.d.usr     %r2, %r3[%r4]
# CHECK: ld.b         %r0, %r1[%r2]      | encoding: [0xf4,0x01,0x1e,0x02]
# CHECK: ld.bu        %r1, %r2[%r3]      | encoding: [0xf4,0x22,0x0e,0x03]
# CHECK: ld.h         %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x1a,0x04]
# CHECK: ld.hu        %r3, %r4[%r5]      | encoding: [0xf4,0x64,0x0a,0x05]
# CHECK: ld           %r4, %r5[%r6]      | encoding: [0xf4,0x85,0x16,0x06]
# CHECK: ld.d         %r5, %r6[%r7]      | encoding: [0xf4,0xa6,0x12,0x07]
# CHECK: ld.b.usr     %r6, %r7[%r8]      | encoding: [0xf4,0xc7,0x1f,0x08]
# CHECK: ld.bu.usr    %r7, %r8[%r9]      | encoding: [0xf4,0xe8,0x0f,0x09]
# CHECK: ld.h.usr     %r8, %r9[%r1]      | encoding: [0xf5,0x09,0x1b,0x01]
# CHECK: ld.hu.usr    %r9, %r1[%r2]      | encoding: [0xf5,0x21,0x0b,0x02]
# CHECK: ld.usr       %r1, %r2[%r3]      | encoding: [0xf4,0x22,0x17,0x03]
# CHECK: ld.d.usr     %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x13,0x04]

# load address
# TODO ld.b %r0, %r1[%r2]
  lda.h        %r0, %r1[%r2]
  lda          %r1, %r2[%r3]
  lda.d        %r2, %r3[%r4]
# CHECK: lda.h        %r0, %r1[%r2]      | encoding: [0xf4,0x01,0x3a,0x02]
# CHECK: lda          %r1, %r2[%r3]      | encoding: [0xf4,0x22,0x36,0x03]
# CHECK: lda.d        %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x32,0x04]

# load from control register
  ldcr         %r0, %cr10
# CHECK: ldcr         %r0, %cr10         | encoding: [0x80,0x00,0x41,0x40]

# make bit field
  mak          %r0, %r1, 10<5>
  mak          %r0, %r1, %r2
  mak          %r0, %r1, 6
  mak          %r0, %r1, <6>
# CHECK: mak      %r0, %r1, 10<5>        | encoding: [0xf0,0x01,0xa1,0x45]
# CHECK: mak      %r0, %r1, %r2          | encoding: [0xf4,0x01,0xa0,0x02]
# CHECK: mak      %r0, %r1, 0<6>         | encoding: [0xf0,0x01,0xa0,0x06]
# CHECK: mak      %r0, %r1, 0<6>         | encoding: [0xf0,0x01,0xa0,0x06]

# logical mask immediate
  mask     %r0, %r1, 0
  mask     %r0, %r1, 4096
  mask.u   %r0, %r1, 0
  mask.u   %r0, %r1, 4096
# CHECK: mask     %r0, %r1, 0            | encoding: [0x48,0x01,0x00,0x00]
# CHECK: mask     %r0, %r1, 4096         | encoding: [0x48,0x01,0x10,0x00]
# CHECK: mask.u   %r0, %r1, 0            | encoding: [0x4c,0x01,0x00,0x00]
# CHECK: mask.u   %r0, %r1, 4096         | encoding: [0x4c,0x01,0x10,0x00]

# integer multiply
  mulu         %r0, %r1, %r2
  mulu         %r0, %r1, 0
  mulu         %r0, %r1, 4096
# CHECK: mulu         %r0, %r1, %r2      | encoding: [0xf4,0x01,0x6c,0x02]
# CHECK: mulu         %r0, %r1, 0        | encoding: [0x6c,0x01,0x00,0x00]
# CHECK: mulu         %r0, %r1, 4096     | encoding: [0x6c,0x01,0x10,0x00]

# floating point round to nearest integer
  nint.ss      %r0, %r10
  nint.sd      %r10, %r12
# CHECK: nint.ss      %r0, %r10          | encoding: [0x84,0x00,0x50,0x0a]
# CHECK: nint.sd      %r10, %r12         | encoding: [0x85,0x40,0x50,0x8c]

# logical or
  or       %r0, %r1, %r2
  or.c     %r1, %r7, %r10
  or       %r0, %r4, 0
  or       %r0, %r4, 4096
  or.u     %r0, %r1, 0
  or.u     %r2, %r4, 4096
# CHECK: or       %r0, %r1, %r2          | encoding: [0xf4,0x01,0x58,0x02]
# CHECK: or.c     %r1, %r7, %r10         | encoding: [0xf4,0x27,0x5c,0x0a]
# CHECK: or       %r0, %r4, 0            | encoding: [0x58,0x04,0x00,0x00]
# CHECK: or       %r0, %r4, 4096         | encoding: [0x58,0x04,0x10,0x00]
# CHECK: or.u     %r0, %r1, 0            | encoding: [0x5c,0x01,0x00,0x00]
# CHECK: or.u     %r2, %r4, 4096         | encoding: [0x5c,0x44,0x10,0x00]

# rotate register
  rot      %r0, %r1, <5>
  rot      %r2, %r4, %r6
# CHECK: rot      %r0, %r1, <5>          | encoding: [0xf0,0x01,0xa8,0x05]
# CHECK: rot      %r2, %r4, %r6          | encoding: [0xf4,0x44,0xa8,0x06]

# return from exception
  rte
# CHECK: rte                             | encoding: [0xf4,0x00,0xfc,0x00]

# set bit field
  set      %r0, %r1, 10<5>
  set      %r2, %r4, %r6
  set      %r3, %r7, 6
  set      %r3, %r7, <6>
# CHECK: set      %r0, %r1, 10<5>        | encoding: [0xf0,0x01,0x89,0x45]
# CHECK: set      %r2, %r4, %r6          | encoding: [0xf4,0x44,0x88,0x06]
# CHECK: set      %r3, %r7, 0<6>         | encoding: [0xf0,0x67,0x88,0x06]
# CHECK: set      %r3, %r7, 0<6>         | encoding: [0xf0,0x67,0x88,0x06]

# store register to memory
  st.b     %r0, %r1, 0
  st.b     %r0, %r1, 4096
  st.h     %r0, %r1, 0
  st.h     %r0, %r1, 4096
  st       %r0, %r1, 0
  st       %r0, %r1, 4096
#  st.d     %r0, %r1, 0
#  st.d     %r0, %r1, 4096
#  st.b     %r0, %r1, %r2
#  st.h     %r2, %r3, %r4
#  st       %r4, %r5, %r6
#  st.d     %r5, %r6, %r7
#  st.b.usr %r6, %r7, %r8
#  st.h.usr %r8, %r9, %r1
#  st.usr   %r1, %r2, %r3
#  st.d.usr %r2, %r3, %r4
#  st.b     %r0, %r1[%r2]
#  st.h     %r2, %r3[%r4]
#  st       %r4, %r5[%r6]
#  st.d     %r5, %r6[%r7]
#  st.b.usr %r6, %r7[%r8]
#  st.h.usr %r8, %r9[%r1]
#  st.usr   %r1, %r2[%r3]
#  st.d.usr %r2, %r3[%r4]
# CHECK: st.b     %r0, %r1, 0            | encoding: [0x2c,0x01,0x00,0x00]
# CHECK: st.b     %r0, %r1, 4096         | encoding: [0x2c,0x01,0x10,0x00]
# CHECK: st.h     %r0, %r1, 0            | encoding: [0x28,0x01,0x00,0x00]
# CHECK: st.h     %r0, %r1, 4096         | encoding: [0x28,0x01,0x10,0x00]
# CHECK: st       %r0, %r1, 0            | encoding: [0x24,0x01,0x00,0x00]
# CHECK: st       %r0, %r1, 4096         | encoding: [0x24,0x01,0x10,0x00]
# COM: CHECK: st.d     %r0, %r1, 0            | encoding: [0x20,0x01,0x00,0x00]
# COM: CHECK: st.d     %r0, %r1, 4096         | encoding: [0x20,0x01,0x10,0x00]
# COM: CHECK: st.b     %r0, %r1, %r2          | encoding: [0xf4,0x01,0x2c,0x02]
# COM: CHECK: st.h     %r2, %r3, %r4          | encoding: [0xf4,0x43,0x28,0x04]
# COM: CHECK: st       %r4, %r5, %r6          | encoding: [0xf4,0x85,0x24,0x06]
# COM: CHECK: st.d     %r5, %r6, %r7          | encoding: [0xf4,0xa6,0x20,0x07]
# COM: CHECK: st.b.usr %r6, %r7, %r8          | encoding: [0xf4,0xc7,0x2d,0x08]
# COM: CHECK: st.h.usr %r8, %r9, %r1          | encoding: [0xf5,0x09,0x29,0x01]
# COM: CHECK: st.usr   %r1, %r2, %r3          | encoding: [0xf4,0x22,0x25,0x03]
# COM: CHECK: st.d.usr %r2, %r3, %r4          | encoding: [0xf4,0x43,0x21,0x04]
# COM: CHECK: st.b     %r0, %r1[%r2]          | encoding: [0xf4,0x01,0x2e,0x02]
# COM: CHECK: st.h     %r2, %r3[%r4]          | encoding: [0xf4,0x43,0x2a,0x04]
# COM: CHECK: st       %r4, %r5[%r6]          | encoding: [0xf4,0x85,0x26,0x06]
# COM: CHECK: st.d     %r5, %r6[%r7]          | encoding: [0xf4,0xa6,0x22,0x07]
# COM: CHECK: st.b.usr %r6, %r7[%r8]          | encoding: [0xf4,0xc7,0x2f,0x08]
# COM: CHECK: st.h.usr %r8, %r9[%r1]          | encoding: [0xf5,0x09,0x2b,0x01]
# COM: CHECK: st.usr   %r1, %r2[%r3]          | encoding: [0xf4,0x22,0x27,0x03]
# COM: CHECK: st.d.usr %r2, %r3[%r4]          | encoding: [0xf4,0x43,0x23,0x04]

# store to control register
  stcr %r0, %cr10
# CHECK: stcr     %r0, %cr10             | encoding: [0x80,0x00,0x81,0x40]

# integer subtract
  sub      %r0, %r1, %r2
  sub.ci   %r1, %r2, %r3
  sub.co   %r2, %r3, %r4
  sub.cio  %r3, %r4, %r5
  sub      %r4, %r5, 0
  sub      %r4, %r5, 4096
# CHECK: sub      %r0, %r1, %r2          | encoding: [0xf4,0x01,0x74,0x02]
# CHECK: sub.ci   %r1, %r2, %r3          | encoding: [0xf4,0x22,0x76,0x03]
# CHECK: sub.co   %r2, %r3, %r4          | encoding: [0xf4,0x43,0x75,0x04]
# CHECK: sub.cio  %r3, %r4, %r5          | encoding: [0xf4,0x64,0x77,0x05]
# CHECK: sub      %r4, %r5, 0            | encoding: [0x74,0x85,0x00,0x00]
# CHECK: sub      %r4, %r5, 4096         | encoding: [0x74,0x85,0x10,0x00]

# unsigned integer subtract
  subu     %r0, %r1, %r2
  subu.ci  %r1, %r2, %r3
  subu.co  %r3, %r4, %r5
  subu.cio %r4, %r5, %r6
  subu     %r5, %r6, 0
  subu     %r5, %r6, 4096
# CHECK: subu     %r0, %r1, %r2          | encoding: [0xf4,0x01,0x64,0x02]
# CHECK: subu.ci  %r1, %r2, %r3          | encoding: [0xf4,0x22,0x66,0x03]
# CHECK: subu.co  %r3, %r4, %r5          | encoding: [0xf4,0x64,0x65,0x05]
# CHECK: subu.cio %r4, %r5, %r6          | encoding: [0xf4,0x85,0x67,0x06]
# CHECK: subu     %r5, %r6, 0            | encoding: [0x64,0xa6,0x00,0x00]
# CHECK: subu     %r5, %r6, 4096         | encoding: [0x64,0xa6,0x10,0x00]

# trap on bit clear
#  tb0          0, %r10, 10
#  tb0          31, %r11, 10
# CHECK: tb0          0, %r10, 10        | encoding: [0xf0,0x0a,0xd0,0x0a]
# CHECK: tb0          31, %r11, 10       | encoding: [0xf3,0xeb,0xd0,0x0a]

# trap on bit set
#  tb1          0, %r10, 10
#  tb1          31, %r11, 10
# CHECK: tb1          0, %r10, 10        | encoding: [0xf0,0x0a,0xd8,0x0a]
# CHECK: tb1          31, %r11, 10       | encoding: [0xf3,0xeb,0xd8,0x0a]

# trap on bounds check
#  tbnd         %r0, %r1
#  tbnd         %r7, 0
#  tbnd         %r7, 4096
# CHECK: tbnd         %r0, %r1           | encoding: [0xf4,0x00,0xf8,0x01]
# CHECK: tbnd         %r7, 0             | encoding: [0xf8,0x07,0x00,0x00]
# CHECK: tbnd         %r7, 4096          | encoding: [0xf8,0x07,0x10,0x00]

# conditional trap
#  tcnd         eq0, %r10, 12
#  tcnd         ne0, %r9, 12
#  tcnd         gt0, %r8, 7
#  tcnd         lt0, %r7, 1
#  tcnd         ge0, %r6, 35
#  tcnd         le0, %r5, 33
#  tcnd         10, %r4, 12
# CHECK: tcnd         eq0, %r10, 12      | encoding: [0xf0,0x4a,0xe8,0x0c]
# CHECK: tcnd         ne0, %r9, 12       | encoding: [0xf1,0xa9,0xe8,0x0c]
# CHECK: tcnd         gt0, %r8, 7        | encoding: [0xf0,0x28,0xe8,0x07]
# CHECK: tcnd         lt0, %r7, 1        | encoding: [0xf1,0x87,0xe8,0x01]
# CHECK: tcnd         ge0, %r6, 35       | encoding: [0xf0,0x66,0xe8,0x23]
# CHECK: tcnd         le0, %r5, 33       | encoding: [0xf1,0xc5,0xe8,0x21]
# CHECK: tcnd         10, %r4, 12        | encoding: [0xf1,0x44,0xe8,0x0c]

# truncate floating point to integer
  trnc.ss      %r0, %r1
  trnc.sd      %r1, %r3
# CHECK: trnc.ss      %r0, %r1           | encoding: [0x84,0x00,0x58,0x01]
# CHECK: trnc.sd      %r1, %r3           | encoding: [0x84,0x20,0x58,0x83]

# exchange control register
  xcr          %r0, %r3, %cr10
# CHECK: xcr          %r0, %r3, %cr10    | encoding: [0x80,0x03,0xc1,0x43]

# exchange register with memory
#  xmem.bu      %r0, %r1, 0
#  xmem.bu      %r0, %r1, 10
#  xmem         %r0, %r1, 0
#  xmem         %r1, %r2, 4096
  xmem.bu      %r0, %r1, %r2
  xmem         %r1, %r2, %r3
  xmem.bu.usr  %r4, %r5, %r6
  xmem.usr     %r5, %r6, %r7
  xmem.bu      %r2, %r3[%r4]
  xmem         %r3, %r4[%r5]
  xmem.bu.usr  %r4, %r5[%r9]
  xmem.usr     %r5, %r6[%r10]
# COM: CHECK: xmem.bu      %r0, %r1, 0        | encoding: [0x]
# COM: CHECK: xmem.bu      %r0, %r1, 10       | encoding: [0x]
# COM: CHECK: xmem         %r0, %r1, 0        | encoding: [0x]
# COM: CHECK: xmem         %r1, %r2, 4096     | encoding: [0x]
# CHECK: xmem.bu      %r0, %r1, %r2      | encoding: [0xf4,0x01,0x00,0x02]
# CHECK: xmem         %r1, %r2, %r3      | encoding: [0xf4,0x22,0x04,0x03]
# CHECK: xmem.bu.usr  %r4, %r5, %r6      | encoding: [0xf4,0x85,0x01,0x06]
# CHECK: xmem.usr     %r5, %r6, %r7      | encoding: [0xf4,0xa6,0x05,0x07]
# CHECK: xmem.bu      %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x02,0x04]
# CHECK: xmem         %r3, %r4[%r5]      | encoding: [0xf4,0x64,0x06,0x05]
# CHECK: xmem.bu.usr  %r4, %r5[%r9]      | encoding: [0xf4,0x85,0x03,0x09]
# CHECK: xmem.usr     %r5, %r6[%r10]     | encoding: [0xf4,0xa6,0x07,0x0a]

# logical exclusive or
  xor      %r0, %r1, %r2
  xor.c    %r1, %r2, %r3
  xor      %r2, %r3, 0
  xor      %r2, %r4, 4096
  xor.u    %r1, %r2, 0
  xor.u    %r2, %r3, 4096
# CHECK: xor      %r0, %r1, %r2          | encoding: [0xf4,0x01,0x50,0x02]
# CHECK: xor.c    %r1, %r2, %r3          | encoding: [0xf4,0x22,0x54,0x03]
# CHECK: xor      %r2, %r3, 0            | encoding: [0x50,0x43,0x00,0x00]
# CHECK: xor      %r2, %r4, 4096         | encoding: [0x50,0x44,0x10,0x00]
# CHECK: xor.u    %r1, %r2, 0            | encoding: [0x54,0x22,0x00,0x00]
# CHECK: xor.u    %r2, %r3, 4096         | encoding: [0x54,0x43,0x10,0x00]
