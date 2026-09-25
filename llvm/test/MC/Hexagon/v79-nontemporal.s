# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv79 -show-encoding < %s \
# RUN:   | FileCheck %s --check-prefix=ENC
#
# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv79 \
# RUN:   -filetype=obj < %s \
# RUN:   | llvm-objdump --mcpu=hexagonv79 -d - \
# RUN:   | FileCheck %s --check-prefix=DIS
#
# Verify raw-byte disassembly independently from assembly.
# RUN: echo "0x00 0xc3 0x02 0xaa" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-B
#
# RUN: echo "0x08 0xc3 0x82 0xaa" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-W
#
# RUN: echo "0x08 0xc2 0xc2 0xaa" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-D
#
# RUN: echo "0x08 0xe3 0x82 0xaa" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-P
#
# RUN: echo "0x0c 0xe3 0x82 0xaa" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-NP
#
# RUN: echo "0x00 0xe0 0xc2 0xa0" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-ZERO
#
# RUN: echo "0x00 0xe0 0x02 0x94" \
# RUN:   | llvm-mc -triple=hexagon -mcpu=hexagonv79 -disassemble \
# RUN:   | FileCheck %s --check-prefix=RAW-FETCH
# RUN: echo 'memw(r2++#4):nt = r3' \
# RUN:   | not llvm-mc -triple=hexagon -mcpu=hexagonv75 2>&1 \
# RUN:   | FileCheck %s --check-prefix=V75-ERR

{
  memb(r2++#0):nt = r3
}
# ENC: memb(r2++#0):nt = r3
# ENC-NEXT: } // encoding: [0x00,0xc3,0x02,0xaa]
# DIS: memb(r2++#0x0):nt = r3

{
  memh(r2++#2):nt = r3
}
# ENC: memh(r2++#2):nt = r3
# ENC-NEXT: } // encoding: [0x08,0xc3,0x42,0xaa]
# DIS: memh(r2++#0x2):nt = r3

{
  memw(r2++#4):nt = r3
}
# ENC: memw(r2++#4):nt = r3
# ENC-NEXT: } // encoding: [0x08,0xc3,0x82,0xaa]
# DIS: memw(r2++#0x4):nt = r3

{
  memd(r2++#8):nt = r3:2
}
# ENC: memd(r2++#8):nt = r3:2
# ENC-NEXT: } // encoding: [0x08,0xc2,0xc2,0xaa]
# DIS: memd(r2++#0x8):nt = r3:2

{
  if (p0) memw(r2++#4):nt = r3
}
# ENC: if (p0) memw(r2++#4):nt = r3
# ENC-NEXT: } // encoding: [0x08,0xe3,0x82,0xaa]
# DIS: if (p0) memw(r2++#0x4):nt = r3

{
  if (!p0) memw(r2++#4):nt = r3
}
# ENC: if (!p0) memw(r2++#4):nt = r3
# ENC-NEXT: } // encoding: [0x0c,0xe3,0x82,0xaa]
# DIS: if (!p0) memw(r2++#0x4):nt = r3

{
  if (p0) memb(r2++#0):nt = r3
}
# ENC: if (p0) memb(r2++#0):nt = r3
# ENC-NEXT: } // encoding: [0x00,0xe3,0x02,0xaa]

{
  if (!p0) memb(r2++#0):nt = r3
}
# ENC: if (!p0) memb(r2++#0):nt = r3
# ENC-NEXT: } // encoding: [0x04,0xe3,0x02,0xaa]

{
  if (p0) memh(r2++#2):nt = r3
}
# ENC: if (p0) memh(r2++#2):nt = r3
# ENC-NEXT: } // encoding: [0x08,0xe3,0x42,0xaa]

{
  if (!p0) memh(r2++#2):nt = r3
}
# ENC: if (!p0) memh(r2++#2):nt = r3
# ENC-NEXT: } // encoding: [0x0c,0xe3,0x42,0xaa]

{
  if (p0) memd(r2++#8):nt = r3:2
}
# ENC: if (p0) memd(r2++#8):nt = r3:2
# ENC-NEXT: } // encoding: [0x08,0xe2,0xc2,0xaa]

{
  if (!p0) memd(r2++#8):nt = r3:2
}
# ENC: if (!p0) memd(r2++#8):nt = r3:2
# ENC-NEXT: } // encoding: [0x0c,0xe2,0xc2,0xaa]

{
  dczeroa(r2):nt
}
# ENC: dczeroa(r2):nt
# ENC-NEXT: } // encoding: [0x00,0xe0,0xc2,0xa0]
# DIS: dczeroa(r2):nt

{
  dcfetch(r2+#0):nt
}
# ENC: dcfetch(r2+#0):nt
# ENC-NEXT: } // encoding: [0x00,0xe0,0x02,0x94]
# DIS: dcfetch(r2+#0x0):nt

# RAW-B: memb(r2++#0):nt = r3
# RAW-W: memw(r2++#4):nt = r3
# RAW-D: memd(r2++#8):nt = r3:2
# RAW-P: if (p0) memw(r2++#4):nt = r3
# RAW-NP: if (!p0) memw(r2++#4):nt = r3
# RAW-ZERO: dczeroa(r2):nt
# RAW-FETCH: dcfetch(r2+#0):nt
# V75-ERR: error: instruction requires: -mv79 or higher
