# RUN: llvm-mc -triple=riscv32 --mattr=+xcvmac -riscv-no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvmac < %s \
# RUN:     | llvm-objdump --mattr=+xcvmac --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.mac t0, t1, t2
# CHECK-INSTR: cv.mac t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mac t0, t1, zero
# CHECK-INSTR: cv.mac t0, t1, zero
# CHECK-ENCODING: [0xab,0x32,0x03,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsn t0, t1, t2, 0
# CHECK-INSTR: cv.machhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsn t0, t1, zero, 16
# CHECK-INSTR: cv.machhsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsn t0, t1, zero, 31
# CHECK-INSTR: cv.machhsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsrn t0, t1, t2, 0
# CHECK-INSTR: cv.machhsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsrn t0, t1, zero, 16
# CHECK-INSTR: cv.machhsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhsrn t0, t1, zero, 31
# CHECK-INSTR: cv.machhsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhun t0, t1, t2, 0
# CHECK-INSTR: cv.machhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhun t0, t1, zero, 16
# CHECK-INSTR: cv.machhun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhun t0, t1, zero, 31
# CHECK-INSTR: cv.machhun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhurn t0, t1, t2, 0
# CHECK-INSTR: cv.machhurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhurn t0, t1, zero, 16
# CHECK-INSTR: cv.machhurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.machhurn t0, t1, zero, 31
# CHECK-INSTR: cv.machhurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsn t0, t1, t2, 0
# CHECK-INSTR: cv.macsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsn t0, t1, zero, 16
# CHECK-INSTR: cv.macsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsn t0, t1, zero, 31
# CHECK-INSTR: cv.macsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsrn t0, t1, t2, 0
# CHECK-INSTR: cv.macsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsrn t0, t1, zero, 16
# CHECK-INSTR: cv.macsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macsrn t0, t1, zero, 31
# CHECK-INSTR: cv.macsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macun t0, t1, t2, 0
# CHECK-INSTR: cv.macun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macun t0, t1, zero, 16
# CHECK-INSTR: cv.macun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macun t0, t1, zero, 31
# CHECK-INSTR: cv.macun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macurn t0, t1, t2, 0
# CHECK-INSTR: cv.macurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macurn t0, t1, zero, 16
# CHECK-INSTR: cv.macurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.macurn t0, t1, zero, 31
# CHECK-INSTR: cv.macurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.msu t0, t1, t2
# CHECK-INSTR: cv.msu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.msu t0, t1, zero
# CHECK-INSTR: cv.msu t0, t1, zero
# CHECK-ENCODING: [0xab,0x32,0x03,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhs t0, t1, t2
# CHECK-INSTR: cv.mulhhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhs t0, t1, zero
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x42,0x03,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsn t0, t1, t2, 0
# CHECK-INSTR: cv.mulhhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsn t0, t1, zero, 16
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsn t0, t1, zero, 31
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsrn t0, t1, t2, 0
# CHECK-INSTR: cv.mulhhsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsrn t0, t1, zero, 16
# CHECK-INSTR: cv.mulhhsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhsrn t0, t1, zero, 31
# CHECK-INSTR: cv.mulhhsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhu t0, t1, t2
# CHECK-INSTR: cv.mulhhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhu t0, t1, zero
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x52,0x03,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhun t0, t1, t2, 0
# CHECK-INSTR: cv.mulhhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}


cv.mulhhun t0, t1, zero, 16
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhun t0, t1, zero, 31
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhurn t0, t1, t2, 0
# CHECK-INSTR: cv.mulhhurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhurn t0, t1, zero, 16
# CHECK-INSTR: cv.mulhhurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulhhurn t0, t1, zero, 31
# CHECK-INSTR: cv.mulhhurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0xfe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.muls t0, t1, t2
# CHECK-INSTR: cv.mulsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.muls t0, t1, zero
# CHECK-INSTR: cv.mulsn t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x42,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsn t0, t1, t2, 0
# CHECK-INSTR: cv.mulsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsn t0, t1, zero, 16
# CHECK-INSTR: cv.mulsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsn t0, t1, zero, 31
# CHECK-INSTR: cv.mulsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsrn t0, t1, t2, 0
# CHECK-INSTR: cv.mulsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsrn t0, t1, zero, 16
# CHECK-INSTR: cv.mulsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulsrn t0, t1, zero, 31
# CHECK-INSTR: cv.mulsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulu t0, t1, t2
# CHECK-INSTR: cv.mulun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulu t0, t1, zero
# CHECK-INSTR: cv.mulun t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x52,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulun t0, t1, t2, 0
# CHECK-INSTR: cv.mulun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulun t0, t1, zero, 16
# CHECK-INSTR: cv.mulun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulun t0, t1, zero, 31
# CHECK-INSTR: cv.mulun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulurn t0, t1, t2, 0
# CHECK-INSTR: cv.mulurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulurn t0, t1, zero, 16
# CHECK-INSTR: cv.mulurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}

cv.mulurn t0, t1, zero, 31
# CHECK-INSTR: cv.mulurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0xbe]
# CHECK-NO-EXT: instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate){{$}}
