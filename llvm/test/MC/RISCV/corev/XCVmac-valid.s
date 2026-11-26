# RUN: llvm-mc -triple=riscv32 --mattr=+xcvmac -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvmac < %s \
# RUN:     | llvm-objdump --mattr=+xcvmac --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.mac t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mac t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x90]

cv.mac t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mac t0, t1, zero
# CHECK-ENCODING: [0xab,0x32,0x03,0x90]

cv.machhsn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x40]

cv.machhsn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0x60]

cv.machhsn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0x7e]

cv.machhsrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0xc0]

cv.machhsrn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0xe0]

cv.machhsrn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0xfe]

cv.machhun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x40]

cv.machhun t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0x60]

cv.machhun t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0x7e]

cv.machhurn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0xc0]

cv.machhurn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0xe0]

cv.machhurn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.machhurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0xfe]

cv.macsn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x00]

cv.macsn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0x20]

cv.macsn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0x3e]

cv.macsrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x62,0x73,0x80]

cv.macsrn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x62,0x03,0xa0]

cv.macsrn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x62,0x03,0xbe]

cv.macun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x00]

cv.macun t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0x20]

cv.macun t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0x3e]

cv.macurn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x72,0x73,0x80]

cv.macurn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x72,0x03,0xa0]

cv.macurn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.macurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x72,0x03,0xbe]

cv.msu t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.msu t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x92]

cv.msu t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.msu t0, t1, zero
# CHECK-ENCODING: [0xab,0x32,0x03,0x92]

cv.mulhhs t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x40]

cv.mulhhs t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x42,0x03,0x40]

cv.mulhhsn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x40]

cv.mulhhsn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0x60]

cv.mulhhsn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0x7e]

cv.mulhhsrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0xc0]

cv.mulhhsrn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0xe0]

cv.mulhhsrn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0xfe]

cv.mulhhu t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x40]

cv.mulhhu t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x52,0x03,0x40]

cv.mulhhun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x40]


cv.mulhhun t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0x60]

cv.mulhhun t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0x7e]

cv.mulhhurn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0xc0]

cv.mulhhurn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0xe0]

cv.mulhhurn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulhhurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0xfe]

cv.muls t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x00]

cv.muls t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsn t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x42,0x03,0x00]

cv.mulsn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x00]

cv.mulsn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0x20]

cv.mulsn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0x3e]

cv.mulsrn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsrn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x42,0x73,0x80]

cv.mulsrn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsrn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x42,0x03,0xa0]

cv.mulsrn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulsrn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x42,0x03,0xbe]

cv.mulu t0, t1, t2 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x00]

cv.mulu t0, t1, zero # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulun t0, t1, zero, 0
# CHECK-ENCODING: [0xdb,0x52,0x03,0x00]

cv.mulun t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulun t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x00]

cv.mulun t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulun t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0x20]

cv.mulun t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulun t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0x3e]

cv.mulurn t0, t1, t2, 0 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulurn t0, t1, t2, 0
# CHECK-ENCODING: [0xdb,0x52,0x73,0x80]

cv.mulurn t0, t1, zero, 16 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulurn t0, t1, zero, 16
# CHECK-ENCODING: [0xdb,0x52,0x03,0xa0]

cv.mulurn t0, t1, zero, 31 # expected-error {{instruction requires the following: 'XCVmac' (CORE-V Multiply-Accumulate)}}
# CHECK-INSTR: cv.mulurn t0, t1, zero, 31
# CHECK-ENCODING: [0xdb,0x52,0x03,0xbe]
