# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v,+xsfvcp %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvcp %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v,+xsfvcp %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvcp --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvcp %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvcp --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v,+xsfvcp %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvcp %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vc.x 0x3, 0xf, 0x1f, a1
# CHECK-INST: sf.vc.x 3, 15, 31, a1
# CHECK-ENCODING: [0xdb,0xcf,0xf5,0x0e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 0ef5cfdb <unknown>

sf.vc.i 0x3, 0xf, 0x1f, 15
# CHECK-INST: sf.vc.i 3, 15, 31, 15
# CHECK-ENCODING: [0xdb,0xbf,0xf7,0x0e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 0ef7bfdb <unknown>

sf.vc.vv 0x3, 0x1f, v2, v1
# CHECK-INST: sf.vc.vv 3, 31, v2, v1
# CHECK-ENCODING: [0xdb,0x8f,0x20,0x2e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2e208fdb <unknown>

sf.vc.xv 0x3, 0x1f, v2, a1
# CHECK-INST: sf.vc.xv 3, 31, v2, a1
# CHECK-ENCODING: [0xdb,0xcf,0x25,0x2e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2e25cfdb <unknown>

sf.vc.iv 0x3, 0x1f, v2, 15
# CHECK-INST: sf.vc.iv 3, 31, v2, 15
# CHECK-ENCODING: [0xdb,0xbf,0x27,0x2e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2e27bfdb <unknown>

sf.vc.fv 0x1, 0x1f, v2, fa1
# CHECK-INST: sf.vc.fv 1, 31, v2, fa1
# CHECK-ENCODING: [0xdb,0xdf,0x25,0x2e]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2e25dfdb <unknown>

sf.vc.vvv 0x3, v0, v2, v1
# CHECK-INST: sf.vc.vvv 3, v0, v2, v1
# CHECK-ENCODING: [0x5b,0x80,0x20,0xae]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ae20805b <unknown>

sf.vc.xvv 0x3, v0, v2, a1
# CHECK-INST: sf.vc.xvv 3, v0, v2, a1
# CHECK-ENCODING: [0x5b,0xc0,0x25,0xae]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ae25c05b <unknown>

sf.vc.ivv 0x3, v0, v2, 15
# CHECK-INST: sf.vc.ivv 3, v0, v2, 15
# CHECK-ENCODING: [0x5b,0xb0,0x27,0xae]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ae27b05b <unknown>

sf.vc.fvv 0x1, v0, v2, fa1
# CHECK-INST: sf.vc.fvv 1, v0, v2, fa1
# CHECK-ENCODING: [0x5b,0xd0,0x25,0xae]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ae25d05b <unknown>

sf.vc.vvw 0x3, v0, v2, v1
# CHECK-INST: sf.vc.vvw 3, v0, v2, v1
# CHECK-ENCODING: [0x5b,0x80,0x20,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fe20805b <unknown>

sf.vc.xvw 0x3, v0, v2, a1
# CHECK-INST: sf.vc.xvw 3, v0, v2, a1
# CHECK-ENCODING: [0x5b,0xc0,0x25,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fe25c05b <unknown>

sf.vc.ivw 0x3, v0, v2, 15
# CHECK-INST: sf.vc.ivw 3, v0, v2, 15
# CHECK-ENCODING: [0x5b,0xb0,0x27,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fe27b05b <unknown>

sf.vc.fvw 0x1, v0, v2, fa1
# CHECK-INST: sf.vc.fvw 1, v0, v2, fa1
# CHECK-ENCODING: [0x5b,0xd0,0x25,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fe25d05b <unknown>

sf.vc.v.x 0x3, 0xf, v0, a1
# CHECK-INST: sf.vc.v.x 3, 15, v0, a1
# CHECK-ENCODING: [0x5b,0xc0,0xf5,0x0c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 0cf5c05b <unknown>

sf.vc.v.i 0x3, 0xf, v0, 15
# CHECK-INST: sf.vc.v.i 3, 15, v0, 15
# CHECK-ENCODING: [0x5b,0xb0,0xf7,0x0c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 0cf7b05b <unknown>

sf.vc.v.vv 0x3, v0, v2, v1
# CHECK-INST: sf.vc.v.vv 3, v0, v2, v1
# CHECK-ENCODING: [0x5b,0x80,0x20,0x2c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2c20805b <unknown>

sf.vc.v.xv 0x3, v0, v2, a1
# CHECK-INST: sf.vc.v.xv 3, v0, v2, a1
# CHECK-ENCODING: [0x5b,0xc0,0x25,0x2c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2c25c05b <unknown>

sf.vc.v.iv 0x3, v0, v2, 15
# CHECK-INST: sf.vc.v.iv 3, v0, v2, 15
# CHECK-ENCODING: [0x5b,0xb0,0x27,0x2c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2c27b05b <unknown>

sf.vc.v.fv 0x1, v0, v2, fa1
# CHECK-INST: sf.vc.v.fv 1, v0, v2, fa1
# CHECK-ENCODING: [0x5b,0xd0,0x25,0x2c]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: 2c25d05b <unknown>

sf.vc.v.vvv 0x3, v0, v2, v1
# CHECK-INST: sf.vc.v.vvv 3, v0, v2, v1
# CHECK-ENCODING: [0x5b,0x80,0x20,0xac]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ac20805b <unknown>

sf.vc.v.xvv 0x3, v0, v2, a1
# CHECK-INST: sf.vc.v.xvv 3, v0, v2, a1
# CHECK-ENCODING: [0x5b,0xc0,0x25,0xac]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ac25c05b <unknown>

sf.vc.v.ivv 0x3, v0, v2, 15
# CHECK-INST: sf.vc.v.ivv 3, v0, v2, 15
# CHECK-ENCODING: [0x5b,0xb0,0x27,0xac]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ac27b05b <unknown>

sf.vc.v.fvv 0x1, v0, v2, fa1
# CHECK-INST: sf.vc.v.fvv 1, v0, v2, fa1
# CHECK-ENCODING: [0x5b,0xd0,0x25,0xac]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: ac25d05b <unknown>

sf.vc.v.vvw 0x3, v0, v2, v1
# CHECK-INST: sf.vc.v.vvw 3, v0, v2, v1
# CHECK-ENCODING: [0x5b,0x80,0x20,0xfc]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fc20805b <unknown>

sf.vc.v.xvw 0x3, v0, v2, a1
# CHECK-INST: sf.vc.v.xvw 3, v0, v2, a1
# CHECK-ENCODING: [0x5b,0xc0,0x25,0xfc]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fc25c05b <unknown>

sf.vc.v.ivw 0x3, v0, v2, 15
# CHECK-INST: sf.vc.v.ivw 3, v0, v2, 15
# CHECK-ENCODING: [0x5b,0xb0,0x27,0xfc]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fc27b05b <unknown>

sf.vc.v.fvw 0x1, v0, v2, fa1
# CHECK-INST: sf.vc.v.fvw 1, v0, v2, fa1
# CHECK-ENCODING: [0x5b,0xd0,0x25,0xfc]
# CHECK-ERROR: instruction requires the following: 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions){{$}}
# CHECK-UNKNOWN: fc25d05b <unknown>
