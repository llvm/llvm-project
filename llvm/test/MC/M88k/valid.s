# RUN: llvm-mc  %s -triple=m88k-unknown-openbsd -show-encoding -mcpu=mc88100 | FileCheck %s

# CHECK: addu     %r0, %r1, %r2          # encoding: [0xf4,0x01,0x60,0x02]
# CHECK: addu.ci  %r1, %r2, %r3          # encoding: [0xf4,0x22,0x62,0x03]
# CHECK: addu.co  %r2, %r3, %r4          # encoding: [0xf4,0x43,0x61,0x04]
# CHECK: addu.cio %r3, %r4, %r5          # encoding: [0xf4,0x64,0x63,0x05]
# CHECK: addu     %r4, %r5, 0            # encoding: [0x60,0x85,0x00,0x00]
# CHECK: addu     %r4, %r5, 4096         # encoding: [0x60,0x85,0x10,0x00]

# CHECK: and      %r0, %r1, %r2          # encoding: [0xf4,0x01,0x40,0x02]
# CHECK: and.c    %r1, %r2, %r3          # encoding: [0xf4,0x22,0x44,0x03]
# CHECK: and      %r2, %r3, 0            # encoding: [0x40,0x43,0x00,0x00]
# CHECK: and      %r2, %r3, 4096         # encoding: [0x40,0x43,0x10,0x00]
# CHECK: and.u    %r2, %r3, 0            # encoding: [0x44,0x43,0x00,0x00]
# CHECK: and.u    %r2, %r3, 4096         # encoding: [0x44,0x43,0x10,0x00]

# CHECK: clr      %r1, %r2, 5<15>        # encoding: [0xf0,0x22,0x80,0xaf]
# CHECK: clr      %r1, %r2, %r3          # encoding: [0xf4,0x22,0x80,0x03]
# CHECK: clr      %r1, %r2, 0<6>         # encoding: [0xf0,0x22,0x80,0x06]
# CHECK: clr      %r1, %r2, 0<6>         # encoding: [0xf0,0x22,0x80,0x06]
# CHECK: cmp      %r0, %r1, %r2          # encoding: [0xf4,0x01,0x7c,0x02]
# COM: CHECK: cmp      %r0, %r2, 0            # encoding: [0x7c,0x02,0x00,0x00]
# COM: CHECK: cmp      %r0, %r2, 4096         # encoding: [0x7c,0x02,0x10,0x00]

# CHECK: ext      %r0, %r1, 10<5>        # encoding: [0xf0,0x01,0x91,0x45]
# CHECK: ext      %r1, %r2, %r3          # encoding: [0xf4,0x22,0x90,0x03]
# CHECK: ext      %r2, %r3, 0<6>         # encoding: [0xf0,0x43,0x90,0x06]
# CHECK: ext      %r2, %r3, 0<6>         # encoding: [0xf0,0x43,0x90,0x06]

# CHECK: extu     %r0, %r1, 10<5>        # encoding: [0xf0,0x01,0x99,0x45]
# CHECK: extu     %r1, %r2, %r3          # encoding: [0xf4,0x22,0x98,0x03]
# CHECK: extu     %r1, %r2, 0<6>         # encoding: [0xf0,0x22,0x98,0x06]
# CHECK: extu     %r1, %r2, 0<6>         # encoding: [0xf0,0x22,0x98,0x06]

# CHECK: ff0      %r1, %r7               # encoding: [0xf4,0x20,0xec,0x07]
# CHECK: ff1      %r3, %r8               # encoding: [0xf4,0x60,0xe8,0x08]

# CHECK: jmp      %r0                    # encoding: [0xf4,0x00,0xc0,0x00]
# CHECK: jmp.n    %r10                   # encoding: [0xf4,0x00,0xc4,0x0a]
# CHECK: jsr      %r10                   # encoding: [0xf4,0x00,0xc8,0x0a]
# CHECK: jsr.n    %r13                   # encoding: [0xf4,0x00,0xcc,0x0d]

# CHECK: ld.b     %r0, %r1, 0            # encoding: [0x1c,0x01,0x00,0x00]
# CHECK: ld.b     %r0, %r1, 4096         # encoding: [0x1c,0x01,0x10,0x00]
# CHECK: ld.bu    %r0, %r1, 0            # encoding: [0x0c,0x01,0x00,0x00]
# CHECK: ld.bu    %r0, %r1, 4096         # encoding: [0x0c,0x01,0x10,0x00]
# CHECK: ld.h     %r0, %r1, 0            # encoding: [0x18,0x01,0x00,0x00]
# CHECK: ld.h     %r0, %r1, 4096         # encoding: [0x18,0x01,0x10,0x00]
# CHECK: ld.hu    %r0, %r1, 0            # encoding: [0x08,0x01,0x00,0x00]
# CHECK: ld.hu    %r0, %r1, 4096         # encoding: [0x08,0x01,0x10,0x00]
# CHECK: ld       %r0, %r1, 0            # encoding: [0x14,0x01,0x00,0x00]
# CHECK: ld       %r0, %r1, 4096         # encoding: [0x14,0x01,0x10,0x00]
# CHECK: ld.d     %r0, %r1, 0            # encoding: [0x10,0x01,0x00,0x00]
# CHECK: ld.d     %r0, %r1, 4096         # encoding: [0x10,0x01,0x10,0x00]

# CHECK: mak      %r0, %r1, 10<5>        # encoding: [0xf0,0x01,0xa1,0x45]
# CHECK: mak      %r0, %r1, %r2          # encoding: [0xf4,0x01,0xa0,0x02]
# CHECK: mak      %r0, %r1, 0<6>         # encoding: [0xf0,0x01,0xa0,0x06]
# CHECK: mak      %r0, %r1, 0<6>         # encoding: [0xf0,0x01,0xa0,0x06]
# CHECK: mask     %r0, %r1, 0            # encoding: [0x48,0x01,0x00,0x00]
# CHECK: mask     %r0, %r1, 4096         # encoding: [0x48,0x01,0x10,0x00]
# CHECK: mask.u   %r0, %r1, 0            # encoding: [0x4c,0x01,0x00,0x00]
# CHECK: mask.u   %r0, %r1, 4096         # encoding: [0x4c,0x01,0x10,0x00]

# CHECK: or       %r0, %r1, %r2          # encoding: [0xf4,0x01,0x58,0x02]
# CHECK: or.c     %r1, %r7, %r10         # encoding: [0xf4,0x27,0x5c,0x0a]
# CHECK: or       %r0, %r4, 0            # encoding: [0x58,0x04,0x00,0x00]
# CHECK: or       %r0, %r4, 4096         # encoding: [0x58,0x04,0x10,0x00]
# CHECK: or.u     %r0, %r1, 0            # encoding: [0x5c,0x01,0x00,0x00]
# CHECK: or.u     %r2, %r4, 4096         # encoding: [0x5c,0x44,0x10,0x00]

# CHECK: rot      %r0, %r1, <5>          # encoding: [0xf0,0x01,0xa8,0x05]
# CHECK: rot      %r2, %r4, %r6          # encoding: [0xf4,0x44,0xa8,0x06]

# CHECK: set      %r0, %r1, 10<5>        # encoding: [0xf0,0x01,0x89,0x45]
# CHECK: set      %r2, %r4, %r6          # encoding: [0xf4,0x44,0x88,0x06]
# CHECK: set      %r3, %r7, 0<6>         # encoding: [0xf0,0x67,0x88,0x06]
# CHECK: set      %r3, %r7, 0<6>         # encoding: [0xf0,0x67,0x88,0x06]
# CHECK: st.b     %r0, %r1, 0            # encoding: [0x2c,0x01,0x00,0x00]
# CHECK: st.b     %r0, %r1, 4096         # encoding: [0x2c,0x01,0x10,0x00]
# CHECK: st.h     %r0, %r1, 0            # encoding: [0x28,0x01,0x00,0x00]
# CHECK: st.h     %r0, %r1, 4096         # encoding: [0x28,0x01,0x10,0x00]
# CHECK: st       %r0, %r1, 0            # encoding: [0x24,0x01,0x00,0x00]
# CHECK: st       %r0, %r1, 4096         # encoding: [0x24,0x01,0x10,0x00]
# CHECK: st.d     %r0, %r1, 0            # encoding: [0x20,0x01,0x00,0x00]
# CHECK: st.d     %r0, %r1, 4096         # encoding: [0x20,0x01,0x10,0x00]
# COM: CHECK: st.b     %r0, %r1, %r2          # encoding: [0xf4,0x01,0x2c,0x02]
# COM: CHECK: st.h     %r2, %r3, %r4          # encoding: [0xf4,0x43,0x28,0x04]
# COM: CHECK: st       %r4, %r5, %r6          # encoding: [0xf4,0x85,0x24,0x06]
# COM: CHECK: st.d     %r5, %r6, %r7          # encoding: [0xf4,0xa6,0x20,0x07]
# COM: CHECK: st.b.usr %r6, %r7, %r8          # encoding: [0xf4,0xc7,0x2d,0x08]
# COM: CHECK: st.h.usr %r8, %r9, %r1          # encoding: [0xf5,0x09,0x29,0x01]
# COM: CHECK: st.usr   %r1, %r2, %r3          # encoding: [0xf4,0x22,0x25,0x03]
# COM: CHECK: st.d.usr %r2, %r3, %r4          # encoding: [0xf4,0x43,0x21,0x04]
# COM: CHECK: st.b     %r0, %r1[%r2]          # encoding: [0xf4,0x01,0x2e,0x02]
# COM: CHECK: st.h     %r2, %r3[%r4]          # encoding: [0xf4,0x43,0x2a,0x04]
# COM: CHECK: st       %r4, %r5[%r6]          # encoding: [0xf4,0x85,0x26,0x06]
# COM: CHECK: st.d     %r5, %r6[%r7]          # encoding: [0xf4,0xa6,0x22,0x07]
# COM: CHECK: st.b.usr %r6, %r7[%r8]          # encoding: [0xf4,0xc7,0x2f,0x08]
# COM: CHECK: st.h.usr %r8, %r9[%r1]          # encoding: [0xf5,0x09,0x2b,0x01]
# COM: CHECK: st.usr   %r1, %r2[%r3]          # encoding: [0xf4,0x22,0x27,0x03]
# COM: CHECK: st.d.usr %r2, %r3[%r4]          # encoding: [0xf4,0x43,0x23,0x04]
# COM: CHECK: stcr     %r0, %cr10             # encoding: [0x80,0x00,0x81,0x40]

# CHECK: subu     %r0, %r1, %r2          # encoding: [0xf4,0x01,0x64,0x02]
# CHECK: subu.ci  %r1, %r2, %r3          # encoding: [0xf4,0x22,0x66,0x03]
# CHECK: subu.co  %r3, %r4, %r5          # encoding: [0xf4,0x64,0x65,0x05]
# CHECK: subu.cio %r4, %r5, %r6          # encoding: [0xf4,0x85,0x67,0x06]
# CHECK: subu     %r5, %r6, 0            # encoding: [0x64,0xa6,0x00,0x00]
# CHECK: subu     %r5, %r6, 4096         # encoding: [0x64,0xa6,0x10,0x00]

# CHECK: xor      %r0, %r1, %r2          # encoding: [0xf4,0x01,0x50,0x02]
# CHECK: xor.c    %r1, %r2, %r3          # encoding: [0xf4,0x22,0x54,0x03]
# CHECK: xor      %r2, %r3, 0            # encoding: [0x50,0x43,0x00,0x00]
# CHECK: xor      %r2, %r4, 4096         # encoding: [0x50,0x44,0x10,0x00]
# CHECK: xor.u    %r1, %r2, 0            # encoding: [0x54,0x22,0x00,0x00]
# CHECK: xor.u    %r2, %r3, 4096         # encoding: [0x54,0x43,0x10,0x00]

foo:
  # unsigned integer add
  addu     %r0, %r1, %r2
  addu.ci  %r1, %r2, %r3
  addu.co  %r2, %r3, %r4
  addu.cio %r3, %r4, %r5
  addu     %r4, %r5, 0
  addu     %r4, %r5, 4096

  # logical and
  and      %r0, %r1, %r2
  and.c    %r1, %r2, %r3
  and      %r2, %r3, 0
  and      %r2, %r3, 4096
  and.u    %r2, %r3, 0
  and.u    %r2, %r3, 4096

  # uncoditional branch
  br       0

  # clear bit field
  clr      %r1, %r2, 5<15>
  clr      %r1, %r2, %r3
  clr      %r1, %r2, 6
  clr      %r1, %r2, <6>

  # integer compare
  cmp      %r0, %r1, %r2
#  cmp      %r0, %r2, 0
#  cmp      %r0, %r2, 4096

  # extract signed bit field
  ext      %r0, %r1, 10<5>
  ext      %r1, %r2, %r3
  ext      %r2, %r3, 6
  ext      %r2, %r3, <6>

  # extract unsigned bit field
  extu     %r0, %r1, 10<5>
  extu     %r1, %r2, %r3
  extu     %r1, %r2, 6
  extu     %r1, %r2, <6>

  # find first bit clear
  ff0      %r1, %r7

  # find first bit set
  ff1      %r3, %r8

  # unconditional jump
  jmp      %r0
  jmp.n    %r10

  # jump to subroutine
  jsr      %r10
  jsr.n    %r13

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

  # make bit field
  mak      %r0, %r1, 10<5>
  mak      %r0, %r1, %r2
  mak      %r0, %r1, 6
  mak      %r0, %r1, <6>

  # logical mask immediate
  mask     %r0, %r1, 0
  mask     %r0, %r1, 4096
  mask.u   %r0, %r1, 0
  mask.u   %r0, %r1, 4096

  # logical or
  or       %r0, %r1, %r2
  or.c     %r1, %r7, %r10
  or       %r0, %r4, 0
  or       %r0, %r4, 4096
  or.u     %r0, %r1, 0
  or.u     %r2, %r4, 4096

  # rotate register
  rot      %r0, %r1, <5>
  rot      %r2, %r4, %r6

  # set bit field
  set      %r0, %r1, 10<5>
  set      %r2, %r4, %r6
  set      %r3, %r7, 6
  set      %r3, %r7, <6>

  # store register to memory
  st.b     %r0, %r1, 0
  st.b     %r0, %r1, 4096
  st.h     %r0, %r1, 0
  st.h     %r0, %r1, 4096
  st       %r0, %r1, 0
  st       %r0, %r1, 4096
  st.d     %r0, %r1, 0
  st.d     %r0, %r1, 4096
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

  # store to control register
#  stcr %r0, %cr10

  # unsigned integer subtract
  subu     %r0, %r1, %r2
  subu.ci  %r1, %r2, %r3
  subu.co  %r3, %r4, %r5
  subu.cio %r4, %r5, %r6
  subu     %r5, %r6, 0
  subu     %r5, %r6, 4096

  # logical exclusive or
  xor      %r0, %r1, %r2
  xor.c    %r1, %r2, %r3
  xor      %r2, %r3, 0
  xor      %r2, %r4, 4096
  xor.u    %r1, %r2, 0
  xor.u    %r2, %r3, 4096
