# RUN: llvm-mc  %s -triple=m88k-unknown-openbsd -show-encoding -mcpu=mc88110 | FileCheck %s
    .requires_88110

# There seems to be a bug in gas. The td and ts2 bits in the floating point
# instructions seems to be wrong. It's not possible to use gas/objdump to verify
# the instructions in this file.

isns:
  # floating point add
  fadd.sss     %x0, %x1, %x2
  fadd.ssd     %x1, %x2, %x3
  fadd.ssx     %x2, %x3, %x4
  fadd.sds     %x3, %x4, %x5
  fadd.sdd     %x4, %x5, %x6
  fadd.sdx     %x5, %x6, %x7
  fadd.sxs     %x6, %x7, %x8
  fadd.sxd     %x7, %x8, %x9
  fadd.sxx     %x8, %x9, %x10
# CHECK: fadd.sss     %x0, %x1, %x2      | encoding: [0x84,0x01,0xa8,0x02]
# CHECK: fadd.ssd     %x1, %x2, %x3      | encoding: [0x84,0x22,0xa8,0x83]
# CHECK: fadd.ssx     %x2, %x3, %x4      | encoding: [0x84,0x43,0xa9,0x04]
# CHECK: fadd.sds     %x3, %x4, %x5      | encoding: [0x84,0x64,0xaa,0x05]
# CHECK: fadd.sdd     %x4, %x5, %x6      | encoding: [0x84,0x85,0xaa,0x86]
# CHECK: fadd.sdx     %x5, %x6, %x7      | encoding: [0x84,0xa6,0xab,0x07]
# CHECK: fadd.sxs     %x6, %x7, %x8      | encoding: [0x84,0xc7,0xac,0x08]
# CHECK: fadd.sxd     %x7, %x8, %x9      | encoding: [0x84,0xe8,0xac,0x89]
# CHECK: fadd.sxx     %x8, %x9, %x10     | encoding: [0x85,0x09,0xad,0x0a]

  fadd.dss     %x9, %x10, %x11
  fadd.dsd     %x10, %x11, %x12
  fadd.dsx     %x11, %x12, %x13
  fadd.dds     %x12, %x13, %x14
  fadd.ddd     %x13, %x14, %x15
  fadd.ddx     %x14, %x15, %x16
  fadd.dxs     %x15, %x16, %x17
  fadd.dxd     %x16, %x17, %x18
  fadd.dxx     %x17, %x18, %x19
# CHECK: fadd.dss     %x9, %x10, %x11    | encoding: [0x85,0x2a,0xa8,0x2b]
# CHECK: fadd.dsd     %x10, %x11, %x12   | encoding: [0x85,0x4b,0xa8,0xac]
# CHECK: fadd.dsx     %x11, %x12, %x13   | encoding: [0x85,0x6c,0xa9,0x2d]
# CHECK: fadd.dds     %x12, %x13, %x14   | encoding: [0x85,0x8d,0xaa,0x2e]
# CHECK: fadd.ddd     %x13, %x14, %x15   | encoding: [0x85,0xae,0xaa,0xaf]
# CHECK: fadd.ddx     %x14, %x15, %x16   | encoding: [0x85,0xcf,0xab,0x30]
# CHECK: fadd.dxs     %x15, %x16, %x17   | encoding: [0x85,0xf0,0xac,0x31]
# CHECK: fadd.dxd     %x16, %x17, %x18   | encoding: [0x86,0x11,0xac,0xb2]
# CHECK: fadd.dxx     %x17, %x18, %x19   | encoding: [0x86,0x32,0xad,0x33]

  fadd.xss     %x18, %x19, %x20
  fadd.xsd     %x19, %x20, %x21
  fadd.xsx     %x20, %x21, %x22
  fadd.xds     %x21, %x22, %x23
  fadd.xdd     %x22, %x23, %x24
  fadd.xdx     %x23, %x24, %x25
  fadd.xxs     %x24, %x25, %x26
  fadd.xxd     %x25, %x26, %x27
  fadd.xxx     %x26, %x27, %x28
# CHECK: fadd.xss     %x18, %x19, %x20   | encoding: [0x86,0x53,0xa8,0x54]
# CHECK: fadd.xsd     %x19, %x20, %x21   | encoding: [0x86,0x74,0xa8,0xd5]
# CHECK: fadd.xsx     %x20, %x21, %x22   | encoding: [0x86,0x95,0xa9,0x56]
# CHECK: fadd.xds     %x21, %x22, %x23   | encoding: [0x86,0xb6,0xaa,0x57]
# CHECK: fadd.xdd     %x22, %x23, %x24   | encoding: [0x86,0xd7,0xaa,0xd8]
# CHECK: fadd.xdx     %x23, %x24, %x25   | encoding: [0x86,0xf8,0xab,0x59]
# CHECK: fadd.xxs     %x24, %x25, %x26   | encoding: [0x87,0x19,0xac,0x5a]
# CHECK: fadd.xxd     %x25, %x26, %x27   | encoding: [0x87,0x3a,0xac,0xdb]
# CHECK: fadd.xxx     %x26, %x27, %x28   | encoding: [0x87,0x5b,0xad,0x5c]

# floating point compare
  fcmp.sss     %r1, %x1, %x2
  fcmp.ssd     %r2, %x3, %x4
  fcmp.ssx     %r3, %x5, %x6
  fcmp.sds     %r4, %x7, %x8
  fcmp.sdd     %r5, %x9, %x10
  fcmp.sdx     %r6, %x11, %x12
  fcmp.sxs     %r7, %x12, %x14
  fcmp.sxd     %r8, %x15, %x16
  fcmp.sxx     %r9, %x17, %x18
# CHECK: fcmp.sss     %r1, %x1, %x2      | encoding: [0x84,0x21,0xb8,0x02]
# CHECK: fcmp.ssd     %r2, %x3, %x4      | encoding: [0x84,0x43,0xb8,0x84]
# CHECK: fcmp.ssx     %r3, %x5, %x6      | encoding: [0x84,0x65,0xb9,0x06]
# CHECK: fcmp.sds     %r4, %x7, %x8      | encoding: [0x84,0x87,0xba,0x08]
# CHECK: fcmp.sdd     %r5, %x9, %x10     | encoding: [0x84,0xa9,0xba,0x8a]
# CHECK: fcmp.sdx     %r6, %x11, %x12    | encoding: [0x84,0xcb,0xbb,0x0c]
# CHECK: fcmp.sxs     %r7, %x12, %x14    | encoding: [0x84,0xec,0xbc,0x0e]
# CHECK: fcmp.sxd     %r8, %x15, %x16    | encoding: [0x85,0x0f,0xbc,0x90]
# CHECK: fcmp.sxx     %r9, %x17, %x18    | encoding: [0x85,0x31,0xbd,0x12]

# unorderd floating point compare
  fcmpu.sss    %r0, %r1, %r2
  fcmpu.ssd    %r0, %r1, %r2
  fcmpu.sds    %r0, %r1, %r2
  fcmpu.sdd    %r0, %r1, %r2
  fcmpu.sss    %r1, %x1, %x2
  fcmpu.ssd    %r2, %x3, %x4
  fcmpu.ssx    %r3, %x5, %x6
  fcmpu.sds    %r4, %x7, %x8
  fcmpu.sdd    %r5, %x9, %x10
  fcmpu.sdx    %r6, %x11, %x12
  fcmpu.sxs    %r7, %x12, %x14
  fcmpu.sxd    %r8, %x15, %x16
  fcmpu.sxx    %r9, %x17, %x18
# CHECK: fcmpu.sss    %r0, %r1, %r2      | encoding: [0x84,0x01,0x38,0x22]
# CHECK: fcmpu.ssd    %r0, %r1, %r2      | encoding: [0x84,0x01,0x38,0xa2]
# CHECK: fcmpu.sds    %r0, %r1, %r2      | encoding: [0x84,0x01,0x3a,0x22]
# CHECK: fcmpu.sdd    %r0, %r1, %r2      | encoding: [0x84,0x01,0x3a,0xa2]
# CHECK: fcmpu.sss    %r1, %x1, %x2      | encoding: [0x84,0x21,0xb8,0x22]
# CHECK: fcmpu.ssd    %r2, %x3, %x4      | encoding: [0x84,0x43,0xb8,0xa4]
# CHECK: fcmpu.ssx    %r3, %x5, %x6      | encoding: [0x84,0x65,0xb9,0x26]
# CHECK: fcmpu.sds    %r4, %x7, %x8      | encoding: [0x84,0x87,0xba,0x28]
# CHECK: fcmpu.sdd    %r5, %x9, %x10     | encoding: [0x84,0xa9,0xba,0xaa]
# CHECK: fcmpu.sdx    %r6, %x11, %x12    | encoding: [0x84,0xcb,0xbb,0x2c]
# CHECK: fcmpu.sxs    %r7, %x12, %x14    | encoding: [0x84,0xec,0xbc,0x2e]
# CHECK: fcmpu.sxd    %r8, %x15, %x16    | encoding: [0x85,0x0f,0xbc,0xb0]
# CHECK: fcmpu.sxx    %r9, %x17, %x18    | encoding: [0x85,0x31,0xbd,0x32]

# floating point conversion
  fcvt.sd      %r1, %r2
  fcvt.ds      %r2, %r1
  fcvt.sd      %x1, %x2
  fcvt.sx      %x1, %x2
  fcvt.ds      %x2, %x3
  fcvt.dx      %x3, %x4
  fcvt.xs      %x4, %x5
  fcvt.xd      %x5, %x6
# CHECK: fcvt.sd      %r1, %r2           | encoding: [0x84,0x20,0x08,0x82]
# CHECK: fcvt.ds      %r2, %r1           | encoding: [0x84,0x40,0x08,0x21]
# CHECK: fcvt.sd      %x1, %x2           | encoding: [0x84,0x20,0x88,0x82]
# CHECK: fcvt.sx      %x1, %x2           | encoding: [0x84,0x20,0x89,0x02]
# CHECK: fcvt.ds      %x2, %x3           | encoding: [0x84,0x40,0x88,0x23]
# CHECK: fcvt.dx      %x3, %x4           | encoding: [0x84,0x60,0x89,0x24]
# CHECK: fcvt.xs      %x4, %x5           | encoding: [0x84,0x80,0x88,0x45]
# CHECK: fcvt.xd      %x5, %x6           | encoding: [0x84,0xa0,0x88,0xc6]

# floating point divide
  fdiv.sss     %x0, %x1, %x2
  fdiv.ssd     %x1, %x2, %x3
  fdiv.ssx     %x2, %x3, %x4
  fdiv.sds     %x3, %x4, %x5
  fdiv.sdd     %x4, %x5, %x6
  fdiv.sdx     %x5, %x6, %x7
  fdiv.sxs     %x6, %x7, %x8
  fdiv.sxd     %x7, %x8, %x9
  fdiv.sxx     %x8, %x9, %x10
# CHECK: fdiv.sss     %x0, %x1, %x2      | encoding: [0x84,0x01,0xf0,0x02]
# CHECK: fdiv.ssd     %x1, %x2, %x3      | encoding: [0x84,0x22,0xf0,0x83]
# CHECK: fdiv.ssx     %x2, %x3, %x4      | encoding: [0x84,0x43,0xf1,0x04]
# CHECK: fdiv.sds     %x3, %x4, %x5      | encoding: [0x84,0x64,0xf2,0x05]
# CHECK: fdiv.sdd     %x4, %x5, %x6      | encoding: [0x84,0x85,0xf2,0x86]
# CHECK: fdiv.sdx     %x5, %x6, %x7      | encoding: [0x84,0xa6,0xf3,0x07]
# CHECK: fdiv.sxs     %x6, %x7, %x8      | encoding: [0x84,0xc7,0xf4,0x08]
# CHECK: fdiv.sxd     %x7, %x8, %x9      | encoding: [0x84,0xe8,0xf4,0x89]
# CHECK: fdiv.sxx     %x8, %x9, %x10     | encoding: [0x85,0x09,0xf5,0x0a]

  fdiv.dss     %x9, %x10, %x11
  fdiv.dsd     %x10, %x11, %x12
  fdiv.dsx     %x11, %x12, %x13
  fdiv.dds     %x12, %x13, %x14
  fdiv.ddd     %x13, %x14, %x15
  fdiv.ddx     %x14, %x15, %x16
  fdiv.dxs     %x15, %x16, %x17
  fdiv.dxd     %x16, %x17, %x18
  fdiv.dxx     %x17, %x18, %x19
# CHECK: fdiv.dss     %x9, %x10, %x11    | encoding: [0x85,0x2a,0xf0,0x2b]
# CHECK: fdiv.dsd     %x10, %x11, %x12   | encoding: [0x85,0x4b,0xf0,0xac]
# CHECK: fdiv.dsx     %x11, %x12, %x13   | encoding: [0x85,0x6c,0xf1,0x2d]
# CHECK: fdiv.dds     %x12, %x13, %x14   | encoding: [0x85,0x8d,0xf2,0x2e]
# CHECK: fdiv.ddd     %x13, %x14, %x15   | encoding: [0x85,0xae,0xf2,0xaf]
# CHECK: fdiv.ddx     %x14, %x15, %x16   | encoding: [0x85,0xcf,0xf3,0x30]
# CHECK: fdiv.dxs     %x15, %x16, %x17   | encoding: [0x85,0xf0,0xf4,0x31]
# CHECK: fdiv.dxd     %x16, %x17, %x18   | encoding: [0x86,0x11,0xf4,0xb2]
# CHECK: fdiv.dxx     %x17, %x18, %x19   | encoding: [0x86,0x32,0xf5,0x33]

  fdiv.xss     %x18, %x19, %x20
  fdiv.xsd     %x19, %x20, %x21
  fdiv.xsx     %x20, %x21, %x22
  fdiv.xds     %x21, %x22, %x23
  fdiv.xdd     %x22, %x23, %x24
  fdiv.xdx     %x23, %x24, %x25
  fdiv.xxs     %x24, %x25, %x26
  fdiv.xxd     %x25, %x26, %x27
  fdiv.xxx     %x26, %x27, %x28
# CHECK: fdiv.xss     %x18, %x19, %x20   | encoding: [0x86,0x53,0xf0,0x54]
# CHECK: fdiv.xsd     %x19, %x20, %x21   | encoding: [0x86,0x74,0xf0,0xd5]
# CHECK: fdiv.xsx     %x20, %x21, %x22   | encoding: [0x86,0x95,0xf1,0x56]
# CHECK: fdiv.xds     %x21, %x22, %x23   | encoding: [0x86,0xb6,0xf2,0x57]
# CHECK: fdiv.xdd     %x22, %x23, %x24   | encoding: [0x86,0xd7,0xf2,0xd8]
# CHECK: fdiv.xdx     %x23, %x24, %x25   | encoding: [0x86,0xf8,0xf3,0x59]
# CHECK: fdiv.xxs     %x24, %x25, %x26   | encoding: [0x87,0x19,0xf4,0x5a]
# CHECK: fdiv.xxd     %x25, %x26, %x27   | encoding: [0x87,0x3a,0xf4,0xdb]
# CHECK: fdiv.xxx     %x26, %x27, %x28   | encoding: [0x87,0x5b,0xf5,0x5c]

# convert integer to floating point
  flt.ss       %x1, %r3
  flt.ds       %x1, %r10
  flt.xs       %x1, %r31
# CHECK: flt.ss       %x1, %r3           | encoding: [0x84,0x20,0x22,0x03]
# CHECK: flt.ds       %x1, %r10          | encoding: [0x84,0x20,0x22,0x2a]
# CHECK: flt.xs       %x1, %r31          | encoding: [0x84,0x20,0x22,0x5f]

# floating point multiply
  fmul.sss     %x0, %x1, %x2
  fmul.ssd     %x1, %x2, %x3
  fmul.ssx     %x2, %x3, %x4
  fmul.sds     %x3, %x4, %x5
  fmul.sdd     %x4, %x5, %x6
  fmul.sdx     %x5, %x6, %x7
  fmul.sxs     %x6, %x7, %x8
  fmul.sxd     %x7, %x8, %x9
  fmul.sxx     %x8, %x9, %x10
# CHECK: fmul.sss     %x0, %x1, %x2      | encoding: [0x84,0x01,0x80,0x02]
# CHECK: fmul.ssd     %x1, %x2, %x3      | encoding: [0x84,0x22,0x80,0x83]
# CHECK: fmul.ssx     %x2, %x3, %x4      | encoding: [0x84,0x43,0x81,0x04]
# CHECK: fmul.sds     %x3, %x4, %x5      | encoding: [0x84,0x64,0x82,0x05]
# CHECK: fmul.sdd     %x4, %x5, %x6      | encoding: [0x84,0x85,0x82,0x86]
# CHECK: fmul.sdx     %x5, %x6, %x7      | encoding: [0x84,0xa6,0x83,0x07]
# CHECK: fmul.sxs     %x6, %x7, %x8      | encoding: [0x84,0xc7,0x84,0x08]
# CHECK: fmul.sxd     %x7, %x8, %x9      | encoding: [0x84,0xe8,0x84,0x89]
# CHECK: fmul.sxx     %x8, %x9, %x10     | encoding: [0x85,0x09,0x85,0x0a]

  fmul.dss     %x9, %x10, %x11
  fmul.dsd     %x10, %x11, %x12
  fmul.dsx     %x11, %x12, %x13
  fmul.dds     %x12, %x13, %x14
  fmul.ddd     %x13, %x14, %x15
  fmul.ddx     %x14, %x15, %x16
  fmul.dxs     %x15, %x16, %x17
  fmul.dxd     %x16, %x17, %x18
  fmul.dxx     %x17, %x18, %x19
# CHECK: fmul.dss     %x9, %x10, %x11    | encoding: [0x85,0x2a,0x80,0x2b]
# CHECK: fmul.dsd     %x10, %x11, %x12   | encoding: [0x85,0x4b,0x80,0xac]
# CHECK: fmul.dsx     %x11, %x12, %x13   | encoding: [0x85,0x6c,0x81,0x2d]
# CHECK: fmul.dds     %x12, %x13, %x14   | encoding: [0x85,0x8d,0x82,0x2e]
# CHECK: fmul.ddd     %x13, %x14, %x15   | encoding: [0x85,0xae,0x82,0xaf]
# CHECK: fmul.ddx     %x14, %x15, %x16   | encoding: [0x85,0xcf,0x83,0x30]
# CHECK: fmul.dxs     %x15, %x16, %x17   | encoding: [0x85,0xf0,0x84,0x31]
# CHECK: fmul.dxd     %x16, %x17, %x18   | encoding: [0x86,0x11,0x84,0xb2]
# CHECK: fmul.dxx     %x17, %x18, %x19   | encoding: [0x86,0x32,0x85,0x33]

  fmul.xss     %x18, %x19, %x20
  fmul.xsd     %x19, %x20, %x21
  fmul.xsx     %x20, %x21, %x22
  fmul.xds     %x21, %x22, %x23
  fmul.xdd     %x22, %x23, %x24
  fmul.xdx     %x23, %x24, %x25
  fmul.xxs     %x24, %x25, %x26
  fmul.xxd     %x25, %x26, %x27
  fmul.xxx     %x26, %x27, %x28
# CHECK: fmul.xss     %x18, %x19, %x20   | encoding: [0x86,0x53,0x80,0x54]
# CHECK: fmul.xsd     %x19, %x20, %x21   | encoding: [0x86,0x74,0x80,0xd5]
# CHECK: fmul.xsx     %x20, %x21, %x22   | encoding: [0x86,0x95,0x81,0x56]
# CHECK: fmul.xds     %x21, %x22, %x23   | encoding: [0x86,0xb6,0x82,0x57]
# CHECK: fmul.xdd     %x22, %x23, %x24   | encoding: [0x86,0xd7,0x82,0xd8]
# CHECK: fmul.xdx     %x23, %x24, %x25   | encoding: [0x86,0xf8,0x83,0x59]
# CHECK: fmul.xxs     %x24, %x25, %x26   | encoding: [0x87,0x19,0x84,0x5a]
# CHECK: fmul.xxd     %x25, %x26, %x27   | encoding: [0x87,0x3a,0x84,0xdb]
# CHECK: fmul.xxx     %x26, %x27, %x28   | encoding: [0x87,0x5b,0x85,0x5c]

# floating point square root
  fsqrt.ss     %r1, %r2
  fsqrt.sd     %r2, %r3
  fsqrt.ds     %r3, %r4
  fsqrt.dd     %r4, %r5
  fsqrt.ss     %x1, %x2
  fsqrt.sd     %x2, %x3
  fsqrt.sx     %x3, %x4
  fsqrt.ds     %x4, %x5
  fsqrt.dd     %x5, %x6
  fsqrt.dx     %x6, %x7
  fsqrt.xs     %x7, %x8
  fsqrt.xd     %x8, %x9
  fsqrt.xx     %x9, %x10
# CHECK: fsqrt.ss     %r1, %r2           | encoding: [0x84,0x20,0x78,0x02]
# CHECK: fsqrt.sd     %r2, %r3           | encoding: [0x84,0x40,0x78,0x83]
# CHECK: fsqrt.ds     %r3, %r4           | encoding: [0x84,0x60,0x78,0x24]
# CHECK: fsqrt.dd     %r4, %r5           | encoding: [0x84,0x80,0x78,0xa5]
# CHECK: fsqrt.ss     %x1, %x2           | encoding: [0x84,0x20,0xf8,0x02]
# CHECK: fsqrt.sd     %x2, %x3           | encoding: [0x84,0x40,0xf8,0x83]
# CHECK: fsqrt.sx     %x3, %x4           | encoding: [0x84,0x60,0xf9,0x04]
# CHECK: fsqrt.ds     %x4, %x5           | encoding: [0x84,0x80,0xf8,0x25]
# CHECK: fsqrt.dd     %x5, %x6           | encoding: [0x84,0xa0,0xf8,0xa6]
# CHECK: fsqrt.dx     %x6, %x7           | encoding: [0x84,0xc0,0xf9,0x27]
# CHECK: fsqrt.xs     %x7, %x8           | encoding: [0x84,0xe0,0xf8,0x48]
# CHECK: fsqrt.xd     %x8, %x9           | encoding: [0x85,0x00,0xf8,0xc9]
# CHECK: fsqrt.xx     %x9, %x10          | encoding: [0x85,0x20,0xf9,0x4a]

# floating point subtract
  fsub.sss     %x0, %x1, %x2
  fsub.ssd     %x1, %x2, %x3
  fsub.ssx     %x2, %x3, %x4
  fsub.sds     %x3, %x4, %x5
  fsub.sdd     %x4, %x5, %x6
  fsub.sdx     %x5, %x6, %x7
  fsub.sxs     %x6, %x7, %x8
  fsub.sxd     %x7, %x8, %x9
  fsub.sxx     %x8, %x9, %x10
# CHECK: fsub.sss     %x0, %x1, %x2      | encoding: [0x84,0x01,0xb0,0x02]
# CHECK: fsub.ssd     %x1, %x2, %x3      | encoding: [0x84,0x22,0xb0,0x83]
# CHECK: fsub.ssx     %x2, %x3, %x4      | encoding: [0x84,0x43,0xb1,0x04]
# CHECK: fsub.sds     %x3, %x4, %x5      | encoding: [0x84,0x64,0xb2,0x05]
# CHECK: fsub.sdd     %x4, %x5, %x6      | encoding: [0x84,0x85,0xb2,0x86]
# CHECK: fsub.sdx     %x5, %x6, %x7      | encoding: [0x84,0xa6,0xb3,0x07]
# CHECK: fsub.sxs     %x6, %x7, %x8      | encoding: [0x84,0xc7,0xb4,0x08]
# CHECK: fsub.sxd     %x7, %x8, %x9      | encoding: [0x84,0xe8,0xb4,0x89]
# CHECK: fsub.sxx     %x8, %x9, %x10     | encoding: [0x85,0x09,0xb5,0x0a]

  fsub.dss     %x9, %x10, %x11
  fsub.dsd     %x10, %x11, %x12
  fsub.dsx     %x11, %x12, %x13
  fsub.dds     %x12, %x13, %x14
  fsub.ddd     %x13, %x14, %x15
  fsub.ddx     %x14, %x15, %x16
  fsub.dxs     %x15, %x16, %x17
  fsub.dxd     %x16, %x17, %x18
  fsub.dxx     %x17, %x18, %x19
# CHECK: fsub.dss     %x9, %x10, %x11    | encoding: [0x85,0x2a,0xb0,0x2b]
# CHECK: fsub.dsd     %x10, %x11, %x12   | encoding: [0x85,0x4b,0xb0,0xac]
# CHECK: fsub.dsx     %x11, %x12, %x13   | encoding: [0x85,0x6c,0xb1,0x2d]
# CHECK: fsub.dds     %x12, %x13, %x14   | encoding: [0x85,0x8d,0xb2,0x2e]
# CHECK: fsub.ddd     %x13, %x14, %x15   | encoding: [0x85,0xae,0xb2,0xaf]
# CHECK: fsub.ddx     %x14, %x15, %x16   | encoding: [0x85,0xcf,0xb3,0x30]
# CHECK: fsub.dxs     %x15, %x16, %x17   | encoding: [0x85,0xf0,0xb4,0x31]
# CHECK: fsub.dxd     %x16, %x17, %x18   | encoding: [0x86,0x11,0xb4,0xb2]
# CHECK: fsub.dxx     %x17, %x18, %x19   | encoding: [0x86,0x32,0xb5,0x33]

  fsub.xss     %x18, %x19, %x20
  fsub.xsd     %x19, %x20, %x21
  fsub.xsx     %x20, %x21, %x22
  fsub.xds     %x21, %x22, %x23
  fsub.xdd     %x22, %x23, %x24
  fsub.xdx     %x23, %x24, %x25
  fsub.xxs     %x24, %x25, %x26
  fsub.xxd     %x25, %x26, %x27
  fsub.xxx     %x26, %x27, %x28
# CHECK: fsub.xss     %x18, %x19, %x20   | encoding: [0x86,0x53,0xb0,0x54]
# CHECK: fsub.xsd     %x19, %x20, %x21   | encoding: [0x86,0x74,0xb0,0xd5]
# CHECK: fsub.xsx     %x20, %x21, %x22   | encoding: [0x86,0x95,0xb1,0x56]
# CHECK: fsub.xds     %x21, %x22, %x23   | encoding: [0x86,0xb6,0xb2,0x57]
# CHECK: fsub.xdd     %x22, %x23, %x24   | encoding: [0x86,0xd7,0xb2,0xd8]
# CHECK: fsub.xdx     %x23, %x24, %x25   | encoding: [0x86,0xf8,0xb3,0x59]
# CHECK: fsub.xxs     %x24, %x25, %x26   | encoding: [0x87,0x19,0xb4,0x5a]
# CHECK: fsub.xxd     %x25, %x26, %x27   | encoding: [0x87,0x3a,0xb4,0xdb]
# CHECK: fsub.xxx     %x26, %x27, %x28   | encoding: [0x87,0x5b,0xb5,0x5c]

# illegal operation
  illop1
  illop2
  illop3
# CHECK: illop1                          | encoding: [0xf4,0x00,0xfc,0x01]
# CHECK: illop2                          | encoding: [0xf4,0x00,0xfc,0x02]
# CHECK: illop3                          | encoding: [0xf4,0x00,0xfc,0x03]

# round floating point to integer
  int.ss       %r1, %x2
  int.sd       %r10, %x3
# CHECK: int.ss       %r1, %x2           | encoding: [0x84,0x20,0xc8,0x02]
# CHECK: int.sd       %r10, %x3          | encoding: [0x85,0x40,0xc8,0x83]

# load register from memory
  ld           %x0, %r1, 0
  ld           %x0, %r1, 4096
  ld.d         %x0, %r1, 0
  ld.d         %x0, %r1, 4096
  ld.x         %x0, %r1, 0
  ld.x         %x0, %r1, 4096
# CHECK: ld           %x0, %r1, 0        | encoding: [0x04,0x01,0x00,0x00]
# CHECK: ld           %x0, %r1, 4096     | encoding: [0x04,0x01,0x10,0x00]
# CHECK: ld.d         %x0, %r1, 0        | encoding: [0x00,0x01,0x00,0x00]
# CHECK: ld.d         %x0, %r1, 4096     | encoding: [0x00,0x01,0x10,0x00]
# CHECK: ld.x         %x0, %r1, 0        | encoding: [0x3c,0x01,0x00,0x00]
# CHECK: ld.x         %x0, %r1, 4096     | encoding: [0x3c,0x01,0x10,0x00]
  ld           %x4, %r5, %r6
  ld.d         %x5, %r6, %r7
  ld.x         %x6, %r7, %r8
  ld.usr       %x1, %r2, %r3
  ld.d.usr     %x2, %r3, %r4
  ld.x.usr     %x3, %r4, %r5
# CHECK: ld           %x4, %r5, %r6      | encoding: [0xf0,0x85,0x14,0x06]
# CHECK: ld.d         %x5, %r6, %r7      | encoding: [0xf0,0xa6,0x10,0x07]
# CHECK: ld.x         %x6, %r7, %r8      | encoding: [0xf0,0xc7,0x18,0x08]
# CHECK: ld.usr       %x1, %r2, %r3      | encoding: [0xf0,0x22,0x15,0x03]
# CHECK: ld.d.usr     %x2, %r3, %r4      | encoding: [0xf0,0x43,0x11,0x04]
# CHECK: ld.x.usr     %x3, %r4, %r5      | encoding: [0xf0,0x64,0x19,0x05]

  ld           %x4, %r5[%r6]
  ld.d         %x5, %r6[%r7]
  ld.x         %x5, %r6[%r7]
  ld.usr       %x4, %r5[%r6]
  ld.d.usr     %x5, %r6[%r7]
  ld.x.usr     %x5, %r6[%r7]

# load address
  lda.x        %r2, %r3[%r4]

# floating point round to nearest integer
  nint.ss      %r1, %x10
  nint.sd      %r10, %x12
# CHECK: nint.ss      %r1, %x10          | encoding: [0x84,0x20,0xd0,0x0a]
# CHECK: nint.sd      %r10, %x12         | encoding: [0x85,0x40,0xd0,0x8c]

# pixel add
  padd.b       %r2, %r4, %r6
  padd.h       %r4, %r6, %r8
  padd         %r6, %r8, %r10
# CHECK: padd.b       %r2, %r4, %r6      | encoding: [0x88,0x44,0x20,0x26]
# CHECK: padd.h       %r4, %r6, %r8      | encoding: [0x88,0x86,0x20,0x48]
# CHECK: padd         %r6, %r8, %r10     | encoding: [0x88,0xc8,0x20,0x6a]

# pixel add and saturate
  padds.u.b    %r2, %r4, %r6
  padds.u.h    %r4, %r6, %r8
  padds.u      %r6, %r8, %r10
  padds.us.b   %r2, %r4, %r6
  padds.us.h   %r4, %r6, %r8
  padds.us     %r6, %r8, %r10
  padds.s.b    %r2, %r4, %r6
  padds.s.h    %r4, %r6, %r8
  padds.s      %r6, %r8, %r10
# CHECK: padds.u.b    %r2, %r4, %r6      | encoding: [0x88,0x44,0x20,0xa6]
# CHECK: padds.u.h    %r4, %r6, %r8      | encoding: [0x88,0x86,0x20,0xc8]
# CHECK: padds.u      %r6, %r8, %r10     | encoding: [0x88,0xc8,0x20,0xea]
# CHECK: padds.us.b   %r2, %r4, %r6      | encoding: [0x88,0x44,0x21,0x26]
# CHECK: padds.us.h   %r4, %r6, %r8      | encoding: [0x88,0x86,0x21,0x48]
# CHECK: padds.us     %r6, %r8, %r10     | encoding: [0x88,0xc8,0x21,0x6a]
# CHECK: padds.s.b    %r2, %r4, %r6      | encoding: [0x88,0x44,0x21,0xa6]
# CHECK: padds.s.h    %r4, %r6, %r8      | encoding: [0x88,0x86,0x21,0xc8]
# CHECK: padds.s      %r6, %r8, %r10     | encoding: [0x88,0xc8,0x21,0xea]

# pixel compare
  pcmp         %r2, %r4, %r6
# CHECK: pcmp         %r2, %r4, %r6      | encoding: [0x88,0x44,0x38,0x66]

# pixel multiply
  pmul         %r2, %r4, %r6
# CHECK: pmul         %r2, %r4, %r6      | encoding: [0x88,0x44,0x00,0x06]

# pixel truncate, insert and pack
  ppack.32.b   %r2, %r4, %r6
  ppack.32.h   %r4, %r6, %r8
  ppack.32     %r6, %r8, %r10
  ppack.16.h   %r4, %r6, %r8
  ppack.16     %r6, %r8, %r10
  ppack.8      %r8, %r10, %r12
# CHECK: ppack.32.b   %r2, %r4, %r6      | encoding: [0x88,0x44,0x61,0x66]
# CHECK: ppack.32.h   %r4, %r6, %r8      | encoding: [0x88,0x86,0x62,0x68]
# CHECK: ppack.32     %r6, %r8, %r10     | encoding: [0x88,0xc8,0x64,0x6a]
# CHECK: ppack.16.h   %r4, %r6, %r8      | encoding: [0x88,0x86,0x62,0x48]
# CHECK: ppack.16     %r6, %r8, %r10     | encoding: [0x88,0xc8,0x64,0x4a]
# CHECK: ppack.8      %r8, %r10, %r12    | encoding: [0x89,0x0a,0x64,0x2c]

# pixel rotate left
  prot         %r2, %r4, %r6
  prot         %r4, %r6, <36>
# CHECK: prot         %r2, %r4, %r6      | encoding: [0x88,0x44,0x78,0x06]
# CHECK: prot         %r4, %r6, <36>     | encoding: [0x88,0x86,0x74,0x80]

# pixel subtract
  psub.b       %r2, %r4, %r6
  psub.h       %r4, %r6, %r8
  psub         %r6, %r8, %r10
# CHECK: psub.b       %r2, %r4, %r6      | encoding: [0x88,0x44,0x30,0x26]
# CHECK: psub.h       %r4, %r6, %r8      | encoding: [0x88,0x86,0x30,0x48]
# CHECK: psub         %r6, %r8, %r10     | encoding: [0x88,0xc8,0x30,0x6a]

# pixel subtract and saturate
  psubs.u.b    %r2, %r4, %r6
  psubs.u.h    %r4, %r6, %r8
  psubs.u      %r6, %r8, %r10
  psubs.us.b   %r2, %r4, %r6
  psubs.us.h   %r4, %r6, %r8
  psubs.us     %r6, %r8, %r10
  psubs.s.b    %r2, %r4, %r6
  psubs.s.h    %r4, %r6, %r8
  psubs.s      %r6, %r8, %r10
# CHECK: psubs.u.b    %r2, %r4, %r6      | encoding: [0x88,0x44,0x30,0xa6]
# CHECK: psubs.u.h    %r4, %r6, %r8      | encoding: [0x88,0x86,0x30,0xc8]
# CHECK: psubs.u      %r6, %r8, %r10     | encoding: [0x88,0xc8,0x30,0xea]
# CHECK: psubs.us.b   %r2, %r4, %r6      | encoding: [0x88,0x44,0x31,0x26]
# CHECK: psubs.us.h   %r4, %r6, %r8      | encoding: [0x88,0x86,0x31,0x48]
# CHECK: psubs.us     %r6, %r8, %r10     | encoding: [0x88,0xc8,0x31,0x6a]
# CHECK: psubs.s.b    %r2, %r4, %r6      | encoding: [0x88,0x44,0x31,0xa6]
# CHECK: psubs.s.h    %r4, %r6, %r8      | encoding: [0x88,0x86,0x31,0xc8]
# CHECK: psubs.s      %r6, %r8, %r10     | encoding: [0x88,0xc8,0x31,0xea]

# pixel unpack
  punpk.n      %r2, %r4
  punpk.b      %r4, %r6
  punpk.h      %r6, %r8
# CHECK: punpk.n      %r2, %r4           | encoding: [0x88,0x44,0x68,0x00]
# CHECK: punpk.b      %r4, %r6           | encoding: [0x88,0x86,0x68,0x20]
# CHECK: punpk.h      %r6, %r8           | encoding: [0x88,0xc8,0x68,0x40]

# store register to memory
  st.b.wt      %r0, %r1, %r2
  st.h.wt      %r2, %r3, %r4
  st.wt        %r4, %r5, %r6
  st.d.wt      %r5, %r6, %r7
  st.b.usr.wt  %r6, %r7, %r8
  st.h.usr.wt  %r8, %r9, %r1
  st.usr.wt    %r1, %r2, %r3
  st.d.usr.wt  %r2, %r3, %r4
  st.b.wt      %r0, %r1[%r2]
  st.h.wt      %r2, %r3[%r4]
  st.wt        %r4, %r5[%r6]
  st.d.wt      %r5, %r6[%r7]
  st.b.usr.wt  %r6, %r7[%r8]
  st.h.usr.wt  %r8, %r9[%r1]
  st.usr.wt    %r1, %r2[%r3]
  st.d.usr.wt  %r2, %r3[%r4]
  st           %x1, %r2, 0
  st           %x1, %r2, 4096
  st.d         %x1, %r2, 0
  st.d         %x1, %r2, 4096
  st.x         %x1, %r2, 0
  st.x         %x1, %r2, 4096
  st           %x0, %r1, %r2
  st.d         %x2, %r3, %r4
  st.x         %x4, %r5, %r6
  st.usr       %x0, %r1, %r2
  st.d.usr     %x2, %r3, %r4
  st.x.usr     %x4, %r5, %r6
  st.wt        %x0, %r1, %r2
  st.d.wt      %x2, %r3, %r4
  st.x.wt      %x4, %r5, %r6
  st.usr.wt    %x0, %r1, %r2
  st.d.usr.wt  %x2, %r3, %r4
  st.x.usr.wt  %x4, %r5, %r6
  st           %x0, %r1[%r2]
  st.d         %x2, %r3[%r4]
  st.x         %x4, %r5[%r6]
  st.usr       %x0, %r1[%r2]
  st.d.usr     %x2, %r3[%r4]
  st.x.usr     %x4, %r5[%r6]
  st.wt        %x0, %r1[%r2]
  st.d.wt      %x2, %r3[%r4]
  st.x.wt      %x4, %r5[%r6]
  st.usr.wt    %x0, %r1[%r2]
  st.d.usr.wt  %x2, %r3[%r4]
  st.x.usr.wt  %x4, %r5[%r6]
# CHECK: st.b.wt      %r0, %r1, %r2      | encoding: [0xf4,0x01,0x2c,0x82]
# CHECK: st.h.wt      %r2, %r3, %r4      | encoding: [0xf4,0x43,0x28,0x84]
# CHECK: st.wt        %r4, %r5, %r6      | encoding: [0xf4,0x85,0x24,0x86]
# CHECK: st.d.wt      %r5, %r6, %r7      | encoding: [0xf4,0xa6,0x20,0x87]
# CHECK: st.b.usr.wt  %r6, %r7, %r8      | encoding: [0xf4,0xc7,0x2d,0x88]
# CHECK: st.h.usr.wt  %r8, %r9, %r1      | encoding: [0xf5,0x09,0x29,0x81]
# CHECK: st.usr.wt    %r1, %r2, %r3      | encoding: [0xf4,0x22,0x25,0x83]
# CHECK: st.d.usr.wt  %r2, %r3, %r4      | encoding: [0xf4,0x43,0x21,0x84]
# CHECK: st.b.wt      %r0, %r1[%r2]      | encoding: [0xf4,0x01,0x2e,0x82]
# CHECK: st.h.wt      %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x2a,0x84]
# CHECK: st.wt        %r4, %r5[%r6]      | encoding: [0xf4,0x85,0x26,0x86]
# CHECK: st.d.wt      %r5, %r6[%r7]      | encoding: [0xf4,0xa6,0x22,0x87]
# CHECK: st.b.usr.wt  %r6, %r7[%r8]      | encoding: [0xf4,0xc7,0x2f,0x88]
# CHECK: st.h.usr.wt  %r8, %r9[%r1]      | encoding: [0xf5,0x09,0x2b,0x81]
# CHECK: st.usr.wt    %r1, %r2[%r3]      | encoding: [0xf4,0x22,0x27,0x83]
# CHECK: st.d.usr.wt  %r2, %r3[%r4]      | encoding: [0xf4,0x43,0x23,0x84]
# CHECK: st           %x1, %r2, 0        | encoding: [0x34,0x22,0x00,0x00]
# CHECK: st           %x1, %r2, 4096     | encoding: [0x34,0x22,0x10,0x00]
# CHECK: st.d         %x1, %r2, 0        | encoding: [0x30,0x22,0x00,0x00]
# CHECK: st.d         %x1, %r2, 4096     | encoding: [0x30,0x22,0x10,0x00]
# CHECK: st.x         %x1, %r2, 0        | encoding: [0x38,0x22,0x00,0x00]
# CHECK: st.x         %x1, %r2, 4096     | encoding: [0x38,0x22,0x10,0x00]
# CHECK: st           %x0, %r1, %r2      | encoding: [0xf0,0x01,0x24,0x02]
# CHECK: st.d         %x2, %r3, %r4      | encoding: [0xf0,0x43,0x20,0x04]
# CHECK: st.x         %x4, %r5, %r6      | encoding: [0xf0,0x85,0x28,0x06]
# CHECK: st.usr       %x0, %r1, %r2      | encoding: [0xf0,0x01,0x25,0x02]
# CHECK: st.d.usr     %x2, %r3, %r4      | encoding: [0xf0,0x43,0x21,0x04]
# CHECK: st.x.usr     %x4, %r5, %r6      | encoding: [0xf0,0x85,0x29,0x06]
# CHECK: st.wt        %x0, %r1, %r2      | encoding: [0xf0,0x01,0x24,0x82]
# CHECK: st.d.wt      %x2, %r3, %r4      | encoding: [0xf0,0x43,0x20,0x84]
# CHECK: st.x.wt      %x4, %r5, %r6      | encoding: [0xf0,0x85,0x28,0x86]
# CHECK: st.usr.wt    %x0, %r1, %r2      | encoding: [0xf0,0x01,0x25,0x82]
# CHECK: st.d.usr.wt  %x2, %r3, %r4      | encoding: [0xf0,0x43,0x21,0x84]
# CHECK: st.x.usr.wt  %x4, %r5, %r6      | encoding: [0xf0,0x85,0x29,0x86]
# CHECK: st           %x0, %r1[%r2]      | encoding: [0xf0,0x01,0x26,0x02]
# CHECK: st.d         %x2, %r3[%r4]      | encoding: [0xf0,0x43,0x22,0x04]
# CHECK: st.x         %x4, %r5[%r6]      | encoding: [0xf0,0x85,0x2a,0x06]
# CHECK: st.usr       %x0, %r1[%r2]      | encoding: [0xf0,0x01,0x27,0x02]
# CHECK: st.d.usr     %x2, %r3[%r4]      | encoding: [0xf0,0x43,0x23,0x04]
# CHECK: st.x.usr     %x4, %r5[%r6]      | encoding: [0xf0,0x85,0x2b,0x06]
# CHECK: st.wt        %x0, %r1[%r2]      | encoding: [0xf0,0x01,0x26,0x82]
# CHECK: st.d.wt      %x2, %r3[%r4]      | encoding: [0xf0,0x43,0x22,0x84]
# CHECK: st.x.wt      %x4, %r5[%r6]      | encoding: [0xf0,0x85,0x2a,0x86]
# CHECK: st.usr.wt    %x0, %r1[%r2]      | encoding: [0xf0,0x01,0x27,0x82]
# CHECK: st.d.usr.wt  %x2, %r3[%r4]      | encoding: [0xf0,0x43,0x23,0x84]
# CHECK: st.x.usr.wt  %x4, %r5[%r6]      | encoding: [0xf0,0x85,0x2b,0x86]

# truncate floating point to integer
  trnc.ss      %r1, %x2
  trnc.sd      %r3, %x4
# CHECK: trnc.ss      %r1, %x2           | encoding: [0x84,0x20,0xd8,0x02]
# CHECK: trnc.sd      %r3, %x4           | encoding: [0x84,0x60,0xd8,0x84]
