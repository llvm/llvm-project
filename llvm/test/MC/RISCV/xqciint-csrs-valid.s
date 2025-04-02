# Xqciint - Qualcomm uC Custom CSRs
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciint -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC  %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciint < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciint -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

csrrs t2, qc.mmcr, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7c]
// CHECK-INST: csrrs t2, qc.mmcr, zero
csrrs t2, 0x7C0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7c]
// CHECK-INST: csrrs t2, qc.mmcr, zero

csrrs t2, qc.mntvec, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7c]
// CHECK-INST: csrrs    t2, qc.mntvec, zero
csrrs t2, 0x7C3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7c]
// CHECK-INST: csrrs    t2, qc.mntvec, zero

csrrs t2, qc.mstktopaddr, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7c]
// CHECK-INST: csrrs    t2, qc.mstktopaddr, zero
csrrs t2, 0x7C4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7c]
// CHECK-INST: csrrs    t2, qc.mstktopaddr, zero

csrrs t2, qc.mstkbottomaddr, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7c]
// CHECK-INST: csrrs    t2, qc.mstkbottomaddr, zero
csrrs t2, 0x7C5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7c]
// CHECK-INST: csrrs    t2, qc.mstkbottomaddr, zero

csrrs t2, qc.mthreadptr, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7c]
// CHECK-INST: csrrs    t2, qc.mthreadptr, zero
csrrs t2, 0x7C8, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7c]
// CHECK-INST: csrrs    t2, qc.mthreadptr, zero

csrrs t2, qc.mcause, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0x7c]
// CHECK-INST: csrrs    t2, qc.mcause, zero
csrrs t2, 0x7C9, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0x7c]
// CHECK-INST: csrrs    t2, qc.mcause, zero

csrrs t2, qc.mclicip0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip0, zero
csrrs t2, 0x7F0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip0, zero

csrrs t2, qc.mclicip1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip1, zero
csrrs t2, 0x7F1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip1, zero

csrrs t2, qc.mclicip2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip2, zero
csrrs t2, 0x7F2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip2, zero

csrrs t2, qc.mclicip3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip3, zero
csrrs t2, 0x7F3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip3, zero

csrrs t2, qc.mclicip4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip4, zero
csrrs t2, 0x7F4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip4, zero

csrrs t2, qc.mclicip5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip5, zero
csrrs t2, 0x7F5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip5, zero

csrrs t2, qc.mclicip6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip6, zero
csrrs t2, 0x7F6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip6, zero

csrrs t2, qc.mclicip7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip7, zero
csrrs t2, 0x7F7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicip7, zero

csrrs t2, qc.mclicie0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie0, zero
csrrs t2, 0x7F8, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie0, zero

csrrs t2, qc.mclicie1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie1, zero
csrrs t2, 0x7F9, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie1, zero

csrrs t2, qc.mclicie2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie2, zero
csrrs t2, 0x7FA, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie2, zero

csrrs t2, qc.mclicie3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie3, zero
csrrs t2, 0x7FB, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie3, zero

csrrs t2, qc.mclicie4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie4, zero
csrrs t2, 0x7FC, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie4, zero

csrrs t2, qc.mclicie5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie5, zero
csrrs t2, 0x7FD, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie5, zero

csrrs t2, qc.mclicie6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie6, zero
csrrs t2, 0x7FE, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie6, zero

csrrs t2, qc.mclicie7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie7, zero
csrrs t2, 0x7FF, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x7f]
// CHECK-INST: csrrs    t2, qc.mclicie7, zero

csrrs t2, qc.mclicilvl00, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl00, zero
csrrs t2, 0xBC0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl00, zero

csrrs t2, qc.mclicilvl01, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl01, zero
csrrs t2, 0xBC1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl01, zero

csrrs t2, qc.mclicilvl02, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl02, zero
csrrs t2, 0xBC2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl02, zero

csrrs t2, qc.mclicilvl03, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl03, zero
csrrs t2, 0xBC3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl03, zero

csrrs t2, qc.mclicilvl04, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl04, zero
csrrs t2, 0xBC4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl04, zero

csrrs t2, qc.mclicilvl05, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl05, zero
csrrs t2, 0xBC5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl05, zero

csrrs t2, qc.mclicilvl06, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl06, zero
csrrs t2, 0xBC6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl06, zero

csrrs t2, qc.mclicilvl07, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl07, zero
csrrs t2, 0xBC7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl07, zero

csrrs t2, qc.mclicilvl08, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl08, zero
csrrs t2, 0xBC8, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl08, zero

csrrs t2, qc.mclicilvl09, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl09, zero
csrrs t2, 0xBC9, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl09, zero

csrrs t2, qc.mclicilvl10, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl10, zero
csrrs t2, 0xBCA, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl10, zero

csrrs t2, qc.mclicilvl11, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl11, zero
csrrs t2, 0xBCB, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl11, zero

csrrs t2, qc.mclicilvl12, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl12, zero
csrrs t2, 0xBCC, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl12, zero

csrrs t2, qc.mclicilvl13, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl13, zero
csrrs t2, 0xBCD, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl13, zero

csrrs t2, qc.mclicilvl14, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl14, zero
csrrs t2, 0xBCE, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl14, zero

csrrs t2, qc.mclicilvl15, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl15, zero
csrrs t2, 0xBCF, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0xbc]
// CHECK-INST: csrrs    t2, qc.mclicilvl15, zero

csrrs t2, qc.mclicilvl16, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl16, zero
csrrs t2, 0xBD0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl16, zero

csrrs t2, qc.mclicilvl17, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl17, zero
csrrs t2, 0xBD1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl17, zero

csrrs t2, qc.mclicilvl18, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl18, zero
csrrs t2, 0xBD2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl18, zero

csrrs t2, qc.mclicilvl19, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl19, zero
csrrs t2, 0xBD3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl19, zero

csrrs t2, qc.mclicilvl20, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl20, zero
csrrs t2, 0xBD4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl20, zero

csrrs t2, qc.mclicilvl21, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl21, zero
csrrs t2, 0xBD5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl21, zero

csrrs t2, qc.mclicilvl22, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl22, zero
csrrs t2, 0xBD6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl22, zero

csrrs t2, qc.mclicilvl23, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl23, zero
csrrs t2, 0xBD7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl23, zero

csrrs t2, qc.mclicilvl24, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl24, zero
csrrs t2, 0xBD8, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x80,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl24, zero

csrrs t2, qc.mclicilvl25, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl25, zero
csrrs t2, 0xBD9, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x90,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl25, zero

csrrs t2, qc.mclicilvl26, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl26, zero
csrrs t2, 0xBDA, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xa0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl26, zero

csrrs t2, qc.mclicilvl27, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl27, zero
csrrs t2, 0xBDB, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xb0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl27, zero

csrrs t2, qc.mclicilvl28, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl28, zero
csrrs t2, 0xBDC, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xc0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl28, zero

csrrs t2, qc.mclicilvl29, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl29, zero
csrrs t2, 0xBDD, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xd0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl29, zero

csrrs t2, qc.mclicilvl30, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl30, zero
csrrs t2, 0xBDE, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xe0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl30, zero

csrrs t2, qc.mclicilvl31, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl31, zero
csrrs t2, 0xBDF, zero
// CHECK-ENC: encoding: [0xf3,0x23,0xf0,0xbd]
// CHECK-INST: csrrs    t2, qc.mclicilvl31, zero

csrrs t2, qc.mwpstartaddr0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr0, zero
csrrs t2, 0x7D0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr0, zero

csrrs t2, qc.mwpstartaddr1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr1, zero
csrrs t2, 0x7D1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr1, zero

csrrs t2, qc.mwpstartaddr2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr2, zero
csrrs t2, 0x7D2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr2, zero

csrrs t2, qc.mwpstartaddr3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr3, zero
csrrs t2, 0x7D3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpstartaddr3, zero

csrrs t2, qc.mwpendaddr0, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr0, zero
csrrs t2, 0x7D4, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr0, zero

csrrs t2, qc.mwpendaddr1, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr1, zero
csrrs t2, 0x7D5, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr1, zero

csrrs t2, qc.mwpendaddr2, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr2, zero
csrrs t2, 0x7D6, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x60,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr2, zero

csrrs t2, qc.mwpendaddr3, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr3, zero
csrrs t2, 0x7D7, zero
// CHECK-ENC: encoding: [0xf3,0x23,0x70,0x7d]
// CHECK-INST: csrrs    t2, qc.mwpendaddr3, zero
