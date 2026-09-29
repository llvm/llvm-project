# RUN: llvm-mc -triple=riscv32 --mattr=+xcvalu -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvalu < %s \
# RUN:     | llvm-objdump --mattr=+xcvalu --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.addurnr t0, t1, t2
# CHECK-INSTR: cv.addurnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addurnr a0, a1, a2
# CHECK-INSTR: cv.addurnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.maxu t0, t1, t2
# CHECK-INSTR: cv.maxu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.maxu a0, a1, a2
# CHECK-INSTR: cv.maxu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subrnr t0, t1, t2
# CHECK-INSTR: cv.subrnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subrnr a0, a1, a2
# CHECK-INSTR: cv.subrnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.sle t0, t1, t2
# CHECK-INSTR: cv.sle t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.slet t0, t1, t2
# CHECK-INSTR: cv.sle t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x52]
# CHECK-NO-EXT: unrecognized instruction mnemonic

cv.sle a0, a1, a2
# CHECK-INSTR: cv.sle a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.slet a0, a1, a2
# CHECK-INSTR: cv.sle a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x52]
# CHECK-NO-EXT: unrecognized instruction mnemonic

cv.subrn t0, t1, t2, 0
# CHECK-INSTR: cv.subrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subrn t0, t1, t2, 16
# CHECK-INSTR: cv.subrn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subrn a0, a1, zero, 31
# CHECK-INSTR: cv.subrn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subnr t0, t1, t2
# CHECK-INSTR: cv.subnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subnr a0, a1, a2
# CHECK-INSTR: cv.subnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addunr t0, t1, t2
# CHECK-INSTR: cv.addunr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addunr a0, a1, a2
# CHECK-INSTR: cv.addunr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addurn t0, t1, t2, 0
# CHECK-INSTR: cv.addurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addurn t0, t1, t2, 16
# CHECK-INSTR: cv.addurn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addurn a0, a1, zero, 31
# CHECK-INSTR: cv.addurn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addun t0, t1, t2, 0
# CHECK-INSTR: cv.addun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addun t0, t1, t2, 16
# CHECK-INSTR: cv.addun t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addun a0, a1, zero, 31
# CHECK-INSTR: cv.addun a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipu t0, t1, 0
# CHECK-INSTR: cv.clipu t0, t1, 0
# CHECK-ENCODING: [0xab,0x32,0x03,0x72]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipu t0, t1, 16
# CHECK-INSTR: cv.clipu t0, t1, 16
# CHECK-ENCODING: [0xab,0x32,0x03,0x73]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipu a0, zero, 31
# CHECK-INSTR: cv.clipu a0, zero, 31
# CHECK-ENCODING: [0x2b,0x35,0xf0,0x73]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clip t0, t1, 0
# CHECK-INSTR: cv.clip t0, t1, 0
# CHECK-ENCODING: [0xab,0x32,0x03,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clip t0, t1, 16
# CHECK-INSTR: cv.clip t0, t1, 16
# CHECK-ENCODING: [0xab,0x32,0x03,0x71]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clip a0, zero, 31
# CHECK-INSTR: cv.clip a0, zero, 31
# CHECK-ENCODING: [0x2b,0x35,0xf0,0x71]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.exthz t0, t1
# CHECK-INSTR: cv.exthz t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x62]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.exthz a0, a1
# CHECK-INSTR: cv.exthz a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x62]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.max t0, t1, t2
# CHECK-INSTR: cv.max t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x5a]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.max a0, a1, a2
# CHECK-INSTR: cv.max a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x5a]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipr t0, t1, t2
# CHECK-INSTR: cv.clipr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipr a0, a1, a2
# CHECK-INSTR: cv.clipr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subn t0, t1, t2, 0
# CHECK-INSTR: cv.subn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subn t0, t1, t2, 16
# CHECK-INSTR: cv.subn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subn a0, a1, zero, 31
# CHECK-INSTR: cv.subn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.extbz t0, t1
# CHECK-INSTR: cv.extbz t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x66]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.extbz a0, a1
# CHECK-INSTR: cv.extbz a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x66]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.abs t0, t1
# CHECK-INSTR: cv.abs t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.abs a0, a1
# CHECK-INSTR: cv.abs a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipur t0, t1, t2
# CHECK-INSTR: cv.clipur t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x76]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.clipur a0, a1, a2
# CHECK-INSTR: cv.clipur a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x76]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.minu t0, t1, t2
# CHECK-INSTR: cv.minu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.minu a0, a1, a2
# CHECK-INSTR: cv.minu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addn t0, t1, t2, 0
# CHECK-INSTR: cv.addn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addn t0, t1, t2, 16
# CHECK-INSTR: cv.addn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addn a0, a1, zero, 31
# CHECK-INSTR: cv.addn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subunr t0, t1, t2
# CHECK-INSTR: cv.subunr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subunr a0, a1, a2
# CHECK-INSTR: cv.subunr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.extbs t0, t1
# CHECK-INSTR: cv.extbs t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.extbs a0, a1
# CHECK-INSTR: cv.extbs a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.sleu t0, t1, t2
# CHECK-INSTR: cv.sleu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.sletu t0, t1, t2
# CHECK-INSTR: cv.sleu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x54]
# CHECK-NO-EXT: unrecognized instruction mnemonic

cv.sleu a0, a1, a2
# CHECK-INSTR: cv.sleu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.sletu a0, a1, a2
# CHECK-INSTR: cv.sleu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x54]
# CHECK-NO-EXT: unrecognized instruction mnemonic

cv.min t0, t1, t2
# CHECK-INSTR: cv.min t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.min a0, a1, a2
# CHECK-INSTR: cv.min a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.suburnr t0, t1, t2
# CHECK-INSTR: cv.suburnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.suburnr a0, a1, a2
# CHECK-INSTR: cv.suburnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addrnr t0, t1, t2
# CHECK-INSTR: cv.addrnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addrnr a0, a1, a2
# CHECK-INSTR: cv.addrnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.exths t0, t1
# CHECK-INSTR: cv.exths t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.exths a0, a1
# CHECK-INSTR: cv.exths a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addrn t0, t1, t2, 0
# CHECK-INSTR: cv.addrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addrn t0, t1, t2, 16
# CHECK-INSTR: cv.addrn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addrn a0, a1, zero, 31
# CHECK-INSTR: cv.addrn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.suburn t0, t1, t2, 0
# CHECK-INSTR: cv.suburn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.suburn t0, t1, t2, 16
# CHECK-INSTR: cv.suburn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.suburn a0, a1, zero, 31
# CHECK-INSTR: cv.suburn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addnr t0, t1, t2
# CHECK-INSTR: cv.addnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.addnr a0, a1, a2
# CHECK-INSTR: cv.addnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subun t0, t1, t2, 0
# CHECK-INSTR: cv.subun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subun t0, t1, t2, 16
# CHECK-INSTR: cv.subun t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}

cv.subun a0, a1, zero, 31
# CHECK-INSTR: cv.subun a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCValu' (CORE-V ALU Operations){{$}}
