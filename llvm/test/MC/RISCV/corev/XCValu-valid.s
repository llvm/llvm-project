# RUN: llvm-mc -triple=riscv32 --mattr=+xcvalu -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvalu < %s \
# RUN:     | llvm-objdump --mattr=+xcvalu --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.addurnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addurnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x86]

cv.addurnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addurnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x86]

cv.maxu t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.maxu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x5c]

cv.maxu a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.maxu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x5c]

cv.subrnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subrnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8c]

cv.subrnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subrnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8c]

cv.sle t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.sle t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x52]

cv.slet t0, t1, t2 # expected-error {{unrecognized instruction mnemonic}}
# CHECK-INSTR: cv.sle t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x52]

cv.sle a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.sle a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x52]

cv.slet a0, a1, a2 # expected-error {{unrecognized instruction mnemonic}}
# CHECK-INSTR: cv.sle a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x52]

cv.subrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x80]

cv.subrn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subrn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0xa0]

cv.subrn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subrn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0xbe]

cv.subnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x88]

cv.subnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x88]

cv.addunr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addunr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x82]

cv.addunr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addunr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x82]

cv.addurn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0xc0]

cv.addurn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addurn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0xe0]

cv.addurn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addurn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0xfe]

cv.addun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x40]

cv.addun t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addun t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0x60]

cv.addun a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addun a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0x7e]

cv.clipu t0, t1, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipu t0, t1, 0
# CHECK-ENCODING: [0xab,0x32,0x03,0x72]

cv.clipu t0, t1, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipu t0, t1, 16
# CHECK-ENCODING: [0xab,0x32,0x03,0x73]

cv.clipu a0, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipu a0, zero, 31
# CHECK-ENCODING: [0x2b,0x35,0xf0,0x73]

cv.clip t0, t1, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clip t0, t1, 0
# CHECK-ENCODING: [0xab,0x32,0x03,0x70]

cv.clip t0, t1, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clip t0, t1, 16
# CHECK-ENCODING: [0xab,0x32,0x03,0x71]

cv.clip a0, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clip a0, zero, 31
# CHECK-ENCODING: [0x2b,0x35,0xf0,0x71]

cv.exthz t0, t1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.exthz t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x62]

cv.exthz a0, a1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.exthz a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x62]

cv.max t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.max t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x5a]

cv.max a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.max a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x5a]

cv.clipr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x74]

cv.clipr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x74]

cv.subn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x00]

cv.subn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0x20]

cv.subn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0x3e]

cv.extbz t0, t1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.extbz t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x66]

cv.extbz a0, a1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.extbz a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x66]

cv.abs t0, t1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.abs t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x50]

cv.abs a0, a1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.abs a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x50]

cv.clipur t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipur t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x76]

cv.clipur a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.clipur a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x76]

cv.minu t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.minu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x58]

cv.minu a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.minu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x58]

cv.addn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x00]

cv.addn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0x20]

cv.addn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0x3e]

cv.subunr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subunr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8a]

cv.subunr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subunr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8a]

cv.extbs t0, t1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.extbs t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x64]

cv.extbs a0, a1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.extbs a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x64]

cv.sleu t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.sleu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x54]

cv.sletu t0, t1, t2 # expected-error {{unrecognized instruction mnemonic}}
# CHECK-INSTR: cv.sleu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x54]

cv.sleu a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.sleu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x54]

cv.sletu a0, a1, a2 # expected-error {{unrecognized instruction mnemonic}}
# CHECK-INSTR: cv.sleu a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x54]

cv.min t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.min t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x56]

cv.min a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.min a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x56]

cv.suburnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.suburnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x8e]

cv.suburnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.suburnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x8e]

cv.addrnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addrnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x84]

cv.addrnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addrnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x84]

cv.exths t0, t1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.exths t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x60]

cv.exths a0, a1 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.exths a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x60]

cv.addrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x22,0x73,0x80]

cv.addrn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addrn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x22,0x73,0xa0]

cv.addrn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addrn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xa5,0x05,0xbe]

cv.suburn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.suburn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0xc0]

cv.suburn t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.suburn t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0xe0]

cv.suburn a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.suburn a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0xfe]

cv.addnr t0, t1, t2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addnr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x80]

cv.addnr a0, a1, a2 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.addnr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x80]

cv.subun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x32,0x73,0x40]

cv.subun t0, t1, t2, 16 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subun t0, t1, t2, 16
# CHECK-ENCODING: [0xdb,0x32,0x73,0x60]

cv.subun a0, a1, zero, 31 # expected-error {{instruction requires the following: 'XCValu' (CORE-V ALU Operations)}}
# CHECK-INSTR: cv.subun a0, a1, zero, 31
# CHECK-ENCODING: [0x5b,0xb5,0x05,0x7e]
