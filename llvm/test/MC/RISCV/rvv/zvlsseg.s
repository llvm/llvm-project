# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvlsseg %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvlsseg %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvlsseg - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvlsseg %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vlseg2e8.v v8, (a0)
# CHECK-INST: vlseg2e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 22 <unknown>

vlseg2e16.v v8, (a0)
# CHECK-INST: vlseg2e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 22 <unknown>

vlseg2e32.v v8, (a0)
# CHECK-INST: vlseg2e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 22 <unknown>

vlseg2e64.v v8, (a0)
# CHECK-INST: vlseg2e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 22 <unknown>

vlseg2e128.v v8, (a0)
# CHECK-INST: vlseg2e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 32 <unknown>

vlseg2e256.v v8, (a0)
# CHECK-INST: vlseg2e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 32 <unknown>

vlseg2e512.v v8, (a0)
# CHECK-INST: vlseg2e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 32 <unknown>

vlseg2e1024.v v8, (a0)
# CHECK-INST: vlseg2e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 32 <unknown>

vlseg2e8.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 20 <unknown>

vlseg2e16.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 20 <unknown>

vlseg2e32.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 20 <unknown>

vlseg2e64.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 20 <unknown>

vlseg2e128.v v8, (a0), v0.t
# CHECK-INST: vlseg2e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 30 <unknown>

vlseg2e256.v v8, (a0), v0.t
# CHECK-INST: vlseg2e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 30 <unknown>

vlseg2e512.v v8, (a0), v0.t
# CHECK-INST: vlseg2e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 30 <unknown>

vlseg2e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg2e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 30 <unknown>

vlseg3e8.v v8, (a0)
# CHECK-INST: vlseg3e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 42 <unknown>

vlseg3e16.v v8, (a0)
# CHECK-INST: vlseg3e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 42 <unknown>

vlseg3e32.v v8, (a0)
# CHECK-INST: vlseg3e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 42 <unknown>

vlseg3e64.v v8, (a0)
# CHECK-INST: vlseg3e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 42 <unknown>

vlseg3e128.v v8, (a0)
# CHECK-INST: vlseg3e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 52 <unknown>

vlseg3e256.v v8, (a0)
# CHECK-INST: vlseg3e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 52 <unknown>

vlseg3e512.v v8, (a0)
# CHECK-INST: vlseg3e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 52 <unknown>

vlseg3e1024.v v8, (a0)
# CHECK-INST: vlseg3e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 52 <unknown>

vlseg3e8.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 40 <unknown>

vlseg3e16.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 40 <unknown>

vlseg3e32.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 40 <unknown>

vlseg3e64.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 40 <unknown>

vlseg3e128.v v8, (a0), v0.t
# CHECK-INST: vlseg3e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 50 <unknown>

vlseg3e256.v v8, (a0), v0.t
# CHECK-INST: vlseg3e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 50 <unknown>

vlseg3e512.v v8, (a0), v0.t
# CHECK-INST: vlseg3e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 50 <unknown>

vlseg3e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg3e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 50 <unknown>

vlseg4e8.v v8, (a0)
# CHECK-INST: vlseg4e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 62 <unknown>

vlseg4e16.v v8, (a0)
# CHECK-INST: vlseg4e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 62 <unknown>

vlseg4e32.v v8, (a0)
# CHECK-INST: vlseg4e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 62 <unknown>

vlseg4e64.v v8, (a0)
# CHECK-INST: vlseg4e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 62 <unknown>

vlseg4e128.v v8, (a0)
# CHECK-INST: vlseg4e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 72 <unknown>

vlseg4e256.v v8, (a0)
# CHECK-INST: vlseg4e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 72 <unknown>

vlseg4e512.v v8, (a0)
# CHECK-INST: vlseg4e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 72 <unknown>

vlseg4e1024.v v8, (a0)
# CHECK-INST: vlseg4e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 72 <unknown>

vlseg4e8.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 60 <unknown>

vlseg4e16.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 60 <unknown>

vlseg4e32.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 60 <unknown>

vlseg4e64.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 60 <unknown>

vlseg4e128.v v8, (a0), v0.t
# CHECK-INST: vlseg4e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 70 <unknown>

vlseg4e256.v v8, (a0), v0.t
# CHECK-INST: vlseg4e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 70 <unknown>

vlseg4e512.v v8, (a0), v0.t
# CHECK-INST: vlseg4e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 70 <unknown>

vlseg4e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg4e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 70 <unknown>

vlseg5e8.v v8, (a0)
# CHECK-INST: vlseg5e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 82 <unknown>

vlseg5e16.v v8, (a0)
# CHECK-INST: vlseg5e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 82 <unknown>

vlseg5e32.v v8, (a0)
# CHECK-INST: vlseg5e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 82 <unknown>

vlseg5e64.v v8, (a0)
# CHECK-INST: vlseg5e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 82 <unknown>

vlseg5e128.v v8, (a0)
# CHECK-INST: vlseg5e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 92 <unknown>

vlseg5e256.v v8, (a0)
# CHECK-INST: vlseg5e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 92 <unknown>

vlseg5e512.v v8, (a0)
# CHECK-INST: vlseg5e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 92 <unknown>

vlseg5e1024.v v8, (a0)
# CHECK-INST: vlseg5e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 92 <unknown>

vlseg5e8.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 80 <unknown>

vlseg5e16.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 80 <unknown>

vlseg5e32.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 80 <unknown>

vlseg5e64.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 80 <unknown>

vlseg5e128.v v8, (a0), v0.t
# CHECK-INST: vlseg5e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 90 <unknown>

vlseg5e256.v v8, (a0), v0.t
# CHECK-INST: vlseg5e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 90 <unknown>

vlseg5e512.v v8, (a0), v0.t
# CHECK-INST: vlseg5e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 90 <unknown>

vlseg5e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg5e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 90 <unknown>

vlseg6e8.v v8, (a0)
# CHECK-INST: vlseg6e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 a2 <unknown>

vlseg6e16.v v8, (a0)
# CHECK-INST: vlseg6e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 a2 <unknown>

vlseg6e32.v v8, (a0)
# CHECK-INST: vlseg6e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 a2 <unknown>

vlseg6e64.v v8, (a0)
# CHECK-INST: vlseg6e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 a2 <unknown>

vlseg6e128.v v8, (a0)
# CHECK-INST: vlseg6e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 b2 <unknown>

vlseg6e256.v v8, (a0)
# CHECK-INST: vlseg6e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 b2 <unknown>

vlseg6e512.v v8, (a0)
# CHECK-INST: vlseg6e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 b2 <unknown>

vlseg6e1024.v v8, (a0)
# CHECK-INST: vlseg6e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 b2 <unknown>

vlseg6e8.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 a0 <unknown>

vlseg6e16.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 a0 <unknown>

vlseg6e32.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 a0 <unknown>

vlseg6e64.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 a0 <unknown>

vlseg6e128.v v8, (a0), v0.t
# CHECK-INST: vlseg6e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 b0 <unknown>

vlseg6e256.v v8, (a0), v0.t
# CHECK-INST: vlseg6e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 b0 <unknown>

vlseg6e512.v v8, (a0), v0.t
# CHECK-INST: vlseg6e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 b0 <unknown>

vlseg6e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg6e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 b0 <unknown>

vlseg7e8.v v8, (a0)
# CHECK-INST: vlseg7e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 c2 <unknown>

vlseg7e16.v v8, (a0)
# CHECK-INST: vlseg7e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 c2 <unknown>

vlseg7e32.v v8, (a0)
# CHECK-INST: vlseg7e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 c2 <unknown>

vlseg7e64.v v8, (a0)
# CHECK-INST: vlseg7e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 c2 <unknown>

vlseg7e128.v v8, (a0)
# CHECK-INST: vlseg7e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 d2 <unknown>

vlseg7e256.v v8, (a0)
# CHECK-INST: vlseg7e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 d2 <unknown>

vlseg7e512.v v8, (a0)
# CHECK-INST: vlseg7e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 d2 <unknown>

vlseg7e1024.v v8, (a0)
# CHECK-INST: vlseg7e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 d2 <unknown>

vlseg7e8.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 c0 <unknown>

vlseg7e16.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 c0 <unknown>

vlseg7e32.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 c0 <unknown>

vlseg7e64.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 c0 <unknown>

vlseg7e128.v v8, (a0), v0.t
# CHECK-INST: vlseg7e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 d0 <unknown>

vlseg7e256.v v8, (a0), v0.t
# CHECK-INST: vlseg7e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 d0 <unknown>

vlseg7e512.v v8, (a0), v0.t
# CHECK-INST: vlseg7e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 d0 <unknown>

vlseg7e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg7e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 d0 <unknown>

vlseg8e8.v v8, (a0)
# CHECK-INST: vlseg8e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 e2 <unknown>

vlseg8e16.v v8, (a0)
# CHECK-INST: vlseg8e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 e2 <unknown>

vlseg8e32.v v8, (a0)
# CHECK-INST: vlseg8e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 e2 <unknown>

vlseg8e64.v v8, (a0)
# CHECK-INST: vlseg8e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 e2 <unknown>

vlseg8e128.v v8, (a0)
# CHECK-INST: vlseg8e128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 f2 <unknown>

vlseg8e256.v v8, (a0)
# CHECK-INST: vlseg8e256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 f2 <unknown>

vlseg8e512.v v8, (a0)
# CHECK-INST: vlseg8e512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 f2 <unknown>

vlseg8e1024.v v8, (a0)
# CHECK-INST: vlseg8e1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 f2 <unknown>

vlseg8e8.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 e0 <unknown>

vlseg8e16.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 e0 <unknown>

vlseg8e32.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 e0 <unknown>

vlseg8e64.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 e0 <unknown>

vlseg8e128.v v8, (a0), v0.t
# CHECK-INST: vlseg8e128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 f0 <unknown>

vlseg8e256.v v8, (a0), v0.t
# CHECK-INST: vlseg8e256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 f0 <unknown>

vlseg8e512.v v8, (a0), v0.t
# CHECK-INST: vlseg8e512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 f0 <unknown>

vlseg8e1024.v v8, (a0), v0.t
# CHECK-INST: vlseg8e1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 f0 <unknown>

vlsseg2e8.v v8, (a0), a1
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 2a <unknown>

vlsseg2e16.v v8, (a0), a1
# CHECK-INST: vlsseg2e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 2a <unknown>

vlsseg2e32.v v8, (a0), a1
# CHECK-INST: vlsseg2e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 2a <unknown>

vlsseg2e64.v v8, (a0), a1
# CHECK-INST: vlsseg2e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 2a <unknown>

vlsseg2e128.v v8, (a0), a1
# CHECK-INST: vlsseg2e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 3a <unknown>

vlsseg2e256.v v8, (a0), a1
# CHECK-INST: vlsseg2e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 3a <unknown>

vlsseg2e512.v v8, (a0), a1
# CHECK-INST: vlsseg2e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 3a <unknown>

vlsseg2e1024.v v8, (a0), a1
# CHECK-INST: vlsseg2e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 3a <unknown>

vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 28 <unknown>

vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 28 <unknown>

vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 28 <unknown>

vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 28 <unknown>

vlsseg2e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 38 <unknown>

vlsseg2e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 38 <unknown>

vlsseg2e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 38 <unknown>

vlsseg2e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 38 <unknown>

vlsseg3e8.v v8, (a0), a1
# CHECK-INST: vlsseg3e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 4a <unknown>

vlsseg3e16.v v8, (a0), a1
# CHECK-INST: vlsseg3e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 4a <unknown>

vlsseg3e32.v v8, (a0), a1
# CHECK-INST: vlsseg3e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 4a <unknown>

vlsseg3e64.v v8, (a0), a1
# CHECK-INST: vlsseg3e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 4a <unknown>

vlsseg3e128.v v8, (a0), a1
# CHECK-INST: vlsseg3e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 5a <unknown>

vlsseg3e256.v v8, (a0), a1
# CHECK-INST: vlsseg3e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 5a <unknown>

vlsseg3e512.v v8, (a0), a1
# CHECK-INST: vlsseg3e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 5a <unknown>

vlsseg3e1024.v v8, (a0), a1
# CHECK-INST: vlsseg3e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 5a <unknown>

vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 48 <unknown>

vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 48 <unknown>

vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 48 <unknown>

vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 48 <unknown>

vlsseg3e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 58 <unknown>

vlsseg3e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 58 <unknown>

vlsseg3e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 58 <unknown>

vlsseg3e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 58 <unknown>

vlsseg4e8.v v8, (a0), a1
# CHECK-INST: vlsseg4e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 6a <unknown>

vlsseg4e16.v v8, (a0), a1
# CHECK-INST: vlsseg4e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 6a <unknown>

vlsseg4e32.v v8, (a0), a1
# CHECK-INST: vlsseg4e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 6a <unknown>

vlsseg4e64.v v8, (a0), a1
# CHECK-INST: vlsseg4e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 6a <unknown>

vlsseg4e128.v v8, (a0), a1
# CHECK-INST: vlsseg4e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 7a <unknown>

vlsseg4e256.v v8, (a0), a1
# CHECK-INST: vlsseg4e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 7a <unknown>

vlsseg4e512.v v8, (a0), a1
# CHECK-INST: vlsseg4e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 7a <unknown>

vlsseg4e1024.v v8, (a0), a1
# CHECK-INST: vlsseg4e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 7a <unknown>

vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 68 <unknown>

vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 68 <unknown>

vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 68 <unknown>

vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 68 <unknown>

vlsseg4e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 78 <unknown>

vlsseg4e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 78 <unknown>

vlsseg4e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 78 <unknown>

vlsseg4e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 78 <unknown>

vlsseg5e8.v v8, (a0), a1
# CHECK-INST: vlsseg5e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 8a <unknown>

vlsseg5e16.v v8, (a0), a1
# CHECK-INST: vlsseg5e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 8a <unknown>

vlsseg5e32.v v8, (a0), a1
# CHECK-INST: vlsseg5e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 8a <unknown>

vlsseg5e64.v v8, (a0), a1
# CHECK-INST: vlsseg5e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 8a <unknown>

vlsseg5e128.v v8, (a0), a1
# CHECK-INST: vlsseg5e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 9a <unknown>

vlsseg5e256.v v8, (a0), a1
# CHECK-INST: vlsseg5e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 9a <unknown>

vlsseg5e512.v v8, (a0), a1
# CHECK-INST: vlsseg5e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 9a <unknown>

vlsseg5e1024.v v8, (a0), a1
# CHECK-INST: vlsseg5e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 9a <unknown>

vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 88 <unknown>

vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 88 <unknown>

vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 88 <unknown>

vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 88 <unknown>

vlsseg5e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 98 <unknown>

vlsseg5e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 98 <unknown>

vlsseg5e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 98 <unknown>

vlsseg5e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 98 <unknown>

vlsseg6e8.v v8, (a0), a1
# CHECK-INST: vlsseg6e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 aa <unknown>

vlsseg6e16.v v8, (a0), a1
# CHECK-INST: vlsseg6e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 aa <unknown>

vlsseg6e32.v v8, (a0), a1
# CHECK-INST: vlsseg6e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 aa <unknown>

vlsseg6e64.v v8, (a0), a1
# CHECK-INST: vlsseg6e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 aa <unknown>

vlsseg6e128.v v8, (a0), a1
# CHECK-INST: vlsseg6e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 ba <unknown>

vlsseg6e256.v v8, (a0), a1
# CHECK-INST: vlsseg6e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 ba <unknown>

vlsseg6e512.v v8, (a0), a1
# CHECK-INST: vlsseg6e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 ba <unknown>

vlsseg6e1024.v v8, (a0), a1
# CHECK-INST: vlsseg6e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 ba <unknown>

vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 a8 <unknown>

vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 a8 <unknown>

vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 a8 <unknown>

vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 a8 <unknown>

vlsseg6e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 b8 <unknown>

vlsseg6e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 b8 <unknown>

vlsseg6e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 b8 <unknown>

vlsseg6e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 b8 <unknown>

vlsseg7e8.v v8, (a0), a1
# CHECK-INST: vlsseg7e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 ca <unknown>

vlsseg7e16.v v8, (a0), a1
# CHECK-INST: vlsseg7e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 ca <unknown>

vlsseg7e32.v v8, (a0), a1
# CHECK-INST: vlsseg7e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 ca <unknown>

vlsseg7e64.v v8, (a0), a1
# CHECK-INST: vlsseg7e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 ca <unknown>

vlsseg7e128.v v8, (a0), a1
# CHECK-INST: vlsseg7e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 da <unknown>

vlsseg7e256.v v8, (a0), a1
# CHECK-INST: vlsseg7e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 da <unknown>

vlsseg7e512.v v8, (a0), a1
# CHECK-INST: vlsseg7e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 da <unknown>

vlsseg7e1024.v v8, (a0), a1
# CHECK-INST: vlsseg7e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 da <unknown>

vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 c8 <unknown>

vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 c8 <unknown>

vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 c8 <unknown>

vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 c8 <unknown>

vlsseg7e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 d8 <unknown>

vlsseg7e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 d8 <unknown>

vlsseg7e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 d8 <unknown>

vlsseg7e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 d8 <unknown>

vlsseg8e8.v v8, (a0), a1
# CHECK-INST: vlsseg8e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 ea <unknown>

vlsseg8e16.v v8, (a0), a1
# CHECK-INST: vlsseg8e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 ea <unknown>

vlsseg8e32.v v8, (a0), a1
# CHECK-INST: vlsseg8e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 ea <unknown>

vlsseg8e64.v v8, (a0), a1
# CHECK-INST: vlsseg8e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 ea <unknown>

vlsseg8e128.v v8, (a0), a1
# CHECK-INST: vlsseg8e128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 fa <unknown>

vlsseg8e256.v v8, (a0), a1
# CHECK-INST: vlsseg8e256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 fa <unknown>

vlsseg8e512.v v8, (a0), a1
# CHECK-INST: vlsseg8e512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 fa <unknown>

vlsseg8e1024.v v8, (a0), a1
# CHECK-INST: vlsseg8e1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 fa <unknown>

vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 e8 <unknown>

vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 e8 <unknown>

vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 e8 <unknown>

vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 e8 <unknown>

vlsseg8e128.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 b5 f8 <unknown>

vlsseg8e256.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 b5 f8 <unknown>

vlsseg8e512.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 b5 f8 <unknown>

vlsseg8e1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 b5 f8 <unknown>

vlxseg2ei8.v v8, (a0), v4
# CHECK-INST: vlxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 2e <unknown>

vlxseg2ei16.v v8, (a0), v4
# CHECK-INST: vlxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 2e <unknown>

vlxseg2ei32.v v8, (a0), v4
# CHECK-INST: vlxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 2e <unknown>

vlxseg2ei64.v v8, (a0), v4
# CHECK-INST: vlxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 2e <unknown>

vlxseg2ei128.v v8, (a0), v4
# CHECK-INST: vlxseg2ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 3e <unknown>

vlxseg2ei256.v v8, (a0), v4
# CHECK-INST: vlxseg2ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 3e <unknown>

vlxseg2ei512.v v8, (a0), v4
# CHECK-INST: vlxseg2ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 3e <unknown>

vlxseg2ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg2ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 3e <unknown>

vlxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 2c <unknown>

vlxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 2c <unknown>

vlxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 2c <unknown>

vlxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 2c <unknown>

vlxseg2ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 3c <unknown>

vlxseg2ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 3c <unknown>

vlxseg2ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 3c <unknown>

vlxseg2ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg2ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 3c <unknown>

vlxseg3ei8.v v8, (a0), v4
# CHECK-INST: vlxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 4e <unknown>

vlxseg3ei16.v v8, (a0), v4
# CHECK-INST: vlxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 4e <unknown>

vlxseg3ei32.v v8, (a0), v4
# CHECK-INST: vlxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 4e <unknown>

vlxseg3ei64.v v8, (a0), v4
# CHECK-INST: vlxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 4e <unknown>

vlxseg3ei128.v v8, (a0), v4
# CHECK-INST: vlxseg3ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 5e <unknown>

vlxseg3ei256.v v8, (a0), v4
# CHECK-INST: vlxseg3ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 5e <unknown>

vlxseg3ei512.v v8, (a0), v4
# CHECK-INST: vlxseg3ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 5e <unknown>

vlxseg3ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg3ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 5e <unknown>

vlxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 4c <unknown>

vlxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 4c <unknown>

vlxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 4c <unknown>

vlxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 4c <unknown>

vlxseg3ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 5c <unknown>

vlxseg3ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 5c <unknown>

vlxseg3ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 5c <unknown>

vlxseg3ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg3ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 5c <unknown>

vlxseg4ei8.v v8, (a0), v4
# CHECK-INST: vlxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 6e <unknown>

vlxseg4ei16.v v8, (a0), v4
# CHECK-INST: vlxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 6e <unknown>

vlxseg4ei32.v v8, (a0), v4
# CHECK-INST: vlxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 6e <unknown>

vlxseg4ei64.v v8, (a0), v4
# CHECK-INST: vlxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 6e <unknown>

vlxseg4ei128.v v8, (a0), v4
# CHECK-INST: vlxseg4ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 7e <unknown>

vlxseg4ei256.v v8, (a0), v4
# CHECK-INST: vlxseg4ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 7e <unknown>

vlxseg4ei512.v v8, (a0), v4
# CHECK-INST: vlxseg4ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 7e <unknown>

vlxseg4ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg4ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 7e <unknown>

vlxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 6c <unknown>

vlxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 6c <unknown>

vlxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 6c <unknown>

vlxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 6c <unknown>

vlxseg4ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 7c <unknown>

vlxseg4ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 7c <unknown>

vlxseg4ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 7c <unknown>

vlxseg4ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg4ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 7c <unknown>

vlxseg5ei8.v v8, (a0), v4
# CHECK-INST: vlxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 8e <unknown>

vlxseg5ei16.v v8, (a0), v4
# CHECK-INST: vlxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 8e <unknown>

vlxseg5ei32.v v8, (a0), v4
# CHECK-INST: vlxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 8e <unknown>

vlxseg5ei64.v v8, (a0), v4
# CHECK-INST: vlxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 8e <unknown>

vlxseg5ei128.v v8, (a0), v4
# CHECK-INST: vlxseg5ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 9e <unknown>

vlxseg5ei256.v v8, (a0), v4
# CHECK-INST: vlxseg5ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 9e <unknown>

vlxseg5ei512.v v8, (a0), v4
# CHECK-INST: vlxseg5ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 9e <unknown>

vlxseg5ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg5ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 9e <unknown>

vlxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 8c <unknown>

vlxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 8c <unknown>

vlxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 8c <unknown>

vlxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 8c <unknown>

vlxseg5ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 9c <unknown>

vlxseg5ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 9c <unknown>

vlxseg5ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 9c <unknown>

vlxseg5ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg5ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 9c <unknown>

vlxseg6ei8.v v8, (a0), v4
# CHECK-INST: vlxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 ae <unknown>

vlxseg6ei16.v v8, (a0), v4
# CHECK-INST: vlxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 ae <unknown>

vlxseg6ei32.v v8, (a0), v4
# CHECK-INST: vlxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 ae <unknown>

vlxseg6ei64.v v8, (a0), v4
# CHECK-INST: vlxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 ae <unknown>

vlxseg6ei128.v v8, (a0), v4
# CHECK-INST: vlxseg6ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 be <unknown>

vlxseg6ei256.v v8, (a0), v4
# CHECK-INST: vlxseg6ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 be <unknown>

vlxseg6ei512.v v8, (a0), v4
# CHECK-INST: vlxseg6ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 be <unknown>

vlxseg6ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg6ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 be <unknown>

vlxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 ac <unknown>

vlxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 ac <unknown>

vlxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 ac <unknown>

vlxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 ac <unknown>

vlxseg6ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 bc <unknown>

vlxseg6ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 bc <unknown>

vlxseg6ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 bc <unknown>

vlxseg6ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg6ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 bc <unknown>

vlxseg7ei8.v v8, (a0), v4
# CHECK-INST: vlxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 ce <unknown>

vlxseg7ei16.v v8, (a0), v4
# CHECK-INST: vlxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 ce <unknown>

vlxseg7ei32.v v8, (a0), v4
# CHECK-INST: vlxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 ce <unknown>

vlxseg7ei64.v v8, (a0), v4
# CHECK-INST: vlxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 ce <unknown>

vlxseg7ei128.v v8, (a0), v4
# CHECK-INST: vlxseg7ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 de <unknown>

vlxseg7ei256.v v8, (a0), v4
# CHECK-INST: vlxseg7ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 de <unknown>

vlxseg7ei512.v v8, (a0), v4
# CHECK-INST: vlxseg7ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 de <unknown>

vlxseg7ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg7ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 de <unknown>

vlxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 cc <unknown>

vlxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 cc <unknown>

vlxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 cc <unknown>

vlxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 cc <unknown>

vlxseg7ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 dc <unknown>

vlxseg7ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 dc <unknown>

vlxseg7ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 dc <unknown>

vlxseg7ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg7ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 dc <unknown>

vlxseg8ei8.v v8, (a0), v4
# CHECK-INST: vlxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 ee <unknown>

vlxseg8ei16.v v8, (a0), v4
# CHECK-INST: vlxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 ee <unknown>

vlxseg8ei32.v v8, (a0), v4
# CHECK-INST: vlxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 ee <unknown>

vlxseg8ei64.v v8, (a0), v4
# CHECK-INST: vlxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 ee <unknown>

vlxseg8ei128.v v8, (a0), v4
# CHECK-INST: vlxseg8ei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 fe <unknown>

vlxseg8ei256.v v8, (a0), v4
# CHECK-INST: vlxseg8ei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 fe <unknown>

vlxseg8ei512.v v8, (a0), v4
# CHECK-INST: vlxseg8ei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 fe <unknown>

vlxseg8ei1024.v v8, (a0), v4
# CHECK-INST: vlxseg8ei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 fe <unknown>

vlxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 ec <unknown>

vlxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 ec <unknown>

vlxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 ec <unknown>

vlxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 ec <unknown>

vlxseg8ei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 45 fc <unknown>

vlxseg8ei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 45 fc <unknown>

vlxseg8ei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 45 fc <unknown>

vlxseg8ei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxseg8ei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 45 fc <unknown>

vlseg2e8ff.v v8, (a0)
# CHECK-INST: vlseg2e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 23 <unknown>

vlseg2e16ff.v v8, (a0)
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 23 <unknown>

vlseg2e32ff.v v8, (a0)
# CHECK-INST: vlseg2e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 23 <unknown>

vlseg2e64ff.v v8, (a0)
# CHECK-INST: vlseg2e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 23 <unknown>

vlseg2e128ff.v v8, (a0)
# CHECK-INST: vlseg2e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 33 <unknown>

vlseg2e256ff.v v8, (a0)
# CHECK-INST: vlseg2e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 33 <unknown>

vlseg2e512ff.v v8, (a0)
# CHECK-INST: vlseg2e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 33 <unknown>

vlseg2e1024ff.v v8, (a0)
# CHECK-INST: vlseg2e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 33 <unknown>

vlseg2e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 21 <unknown>

vlseg2e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 21 <unknown>

vlseg2e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 21 <unknown>

vlseg2e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 21 <unknown>

vlseg2e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 31 <unknown>

vlseg2e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 31 <unknown>

vlseg2e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 31 <unknown>

vlseg2e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 31 <unknown>

vlseg3e8ff.v v8, (a0)
# CHECK-INST: vlseg3e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 43 <unknown>

vlseg3e16ff.v v8, (a0)
# CHECK-INST: vlseg3e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 43 <unknown>

vlseg3e32ff.v v8, (a0)
# CHECK-INST: vlseg3e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 43 <unknown>

vlseg3e64ff.v v8, (a0)
# CHECK-INST: vlseg3e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 43 <unknown>

vlseg3e128ff.v v8, (a0)
# CHECK-INST: vlseg3e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 53 <unknown>

vlseg3e256ff.v v8, (a0)
# CHECK-INST: vlseg3e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 53 <unknown>

vlseg3e512ff.v v8, (a0)
# CHECK-INST: vlseg3e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 53 <unknown>

vlseg3e1024ff.v v8, (a0)
# CHECK-INST: vlseg3e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 53 <unknown>

vlseg3e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 41 <unknown>

vlseg3e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 41 <unknown>

vlseg3e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 41 <unknown>

vlseg3e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 41 <unknown>

vlseg3e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 51 <unknown>

vlseg3e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 51 <unknown>

vlseg3e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 51 <unknown>

vlseg3e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 51 <unknown>

vlseg4e8ff.v v8, (a0)
# CHECK-INST: vlseg4e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 63 <unknown>

vlseg4e16ff.v v8, (a0)
# CHECK-INST: vlseg4e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 63 <unknown>

vlseg4e32ff.v v8, (a0)
# CHECK-INST: vlseg4e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 63 <unknown>

vlseg4e64ff.v v8, (a0)
# CHECK-INST: vlseg4e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 63 <unknown>

vlseg4e128ff.v v8, (a0)
# CHECK-INST: vlseg4e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 73 <unknown>

vlseg4e256ff.v v8, (a0)
# CHECK-INST: vlseg4e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 73 <unknown>

vlseg4e512ff.v v8, (a0)
# CHECK-INST: vlseg4e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 73 <unknown>

vlseg4e1024ff.v v8, (a0)
# CHECK-INST: vlseg4e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 73 <unknown>

vlseg4e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 61 <unknown>

vlseg4e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 61 <unknown>

vlseg4e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 61 <unknown>

vlseg4e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 61 <unknown>

vlseg4e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 71 <unknown>

vlseg4e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 71 <unknown>

vlseg4e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 71 <unknown>

vlseg4e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 71 <unknown>

vlseg5e8ff.v v8, (a0)
# CHECK-INST: vlseg5e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 83 <unknown>

vlseg5e16ff.v v8, (a0)
# CHECK-INST: vlseg5e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 83 <unknown>

vlseg5e32ff.v v8, (a0)
# CHECK-INST: vlseg5e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 83 <unknown>

vlseg5e64ff.v v8, (a0)
# CHECK-INST: vlseg5e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 83 <unknown>

vlseg5e128ff.v v8, (a0)
# CHECK-INST: vlseg5e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 93 <unknown>

vlseg5e256ff.v v8, (a0)
# CHECK-INST: vlseg5e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 93 <unknown>

vlseg5e512ff.v v8, (a0)
# CHECK-INST: vlseg5e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 93 <unknown>

vlseg5e1024ff.v v8, (a0)
# CHECK-INST: vlseg5e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 93 <unknown>

vlseg5e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 81 <unknown>

vlseg5e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 81 <unknown>

vlseg5e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 81 <unknown>

vlseg5e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 81 <unknown>

vlseg5e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 91 <unknown>

vlseg5e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 91 <unknown>

vlseg5e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 91 <unknown>

vlseg5e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 91 <unknown>

vlseg6e8ff.v v8, (a0)
# CHECK-INST: vlseg6e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 a3 <unknown>

vlseg6e16ff.v v8, (a0)
# CHECK-INST: vlseg6e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 a3 <unknown>

vlseg6e32ff.v v8, (a0)
# CHECK-INST: vlseg6e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 a3 <unknown>

vlseg6e64ff.v v8, (a0)
# CHECK-INST: vlseg6e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 a3 <unknown>

vlseg6e128ff.v v8, (a0)
# CHECK-INST: vlseg6e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 b3 <unknown>

vlseg6e256ff.v v8, (a0)
# CHECK-INST: vlseg6e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 b3 <unknown>

vlseg6e512ff.v v8, (a0)
# CHECK-INST: vlseg6e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 b3 <unknown>

vlseg6e1024ff.v v8, (a0)
# CHECK-INST: vlseg6e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 b3 <unknown>

vlseg6e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 a1 <unknown>

vlseg6e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 a1 <unknown>

vlseg6e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 a1 <unknown>

vlseg6e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 a1 <unknown>

vlseg6e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 b1 <unknown>

vlseg6e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 b1 <unknown>

vlseg6e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 b1 <unknown>

vlseg6e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 b1 <unknown>

vlseg7e8ff.v v8, (a0)
# CHECK-INST: vlseg7e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 c3 <unknown>

vlseg7e16ff.v v8, (a0)
# CHECK-INST: vlseg7e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 c3 <unknown>

vlseg7e32ff.v v8, (a0)
# CHECK-INST: vlseg7e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 c3 <unknown>

vlseg7e64ff.v v8, (a0)
# CHECK-INST: vlseg7e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 c3 <unknown>

vlseg7e128ff.v v8, (a0)
# CHECK-INST: vlseg7e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 d3 <unknown>

vlseg7e256ff.v v8, (a0)
# CHECK-INST: vlseg7e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 d3 <unknown>

vlseg7e512ff.v v8, (a0)
# CHECK-INST: vlseg7e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 d3 <unknown>

vlseg7e1024ff.v v8, (a0)
# CHECK-INST: vlseg7e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 d3 <unknown>

vlseg7e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 c1 <unknown>

vlseg7e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 c1 <unknown>

vlseg7e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 c1 <unknown>

vlseg7e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 c1 <unknown>

vlseg7e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 d1 <unknown>

vlseg7e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 d1 <unknown>

vlseg7e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 d1 <unknown>

vlseg7e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 d1 <unknown>

vlseg8e8ff.v v8, (a0)
# CHECK-INST: vlseg8e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 e3 <unknown>

vlseg8e16ff.v v8, (a0)
# CHECK-INST: vlseg8e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 e3 <unknown>

vlseg8e32ff.v v8, (a0)
# CHECK-INST: vlseg8e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 e3 <unknown>

vlseg8e64ff.v v8, (a0)
# CHECK-INST: vlseg8e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 e3 <unknown>

vlseg8e128ff.v v8, (a0)
# CHECK-INST: vlseg8e128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 f3 <unknown>

vlseg8e256ff.v v8, (a0)
# CHECK-INST: vlseg8e256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 f3 <unknown>

vlseg8e512ff.v v8, (a0)
# CHECK-INST: vlseg8e512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 f3 <unknown>

vlseg8e1024ff.v v8, (a0)
# CHECK-INST: vlseg8e1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 f3 <unknown>

vlseg8e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 e1 <unknown>

vlseg8e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 e1 <unknown>

vlseg8e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 e1 <unknown>

vlseg8e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 e1 <unknown>

vlseg8e128ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 04 05 f1 <unknown>

vlseg8e256ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 54 05 f1 <unknown>

vlseg8e512ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 64 05 f1 <unknown>

vlseg8e1024ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 07 74 05 f1 <unknown>

vsseg2e8.v v24, (a0)
# CHECK-INST: vsseg2e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 22 <unknown>

vsseg2e16.v v24, (a0)
# CHECK-INST: vsseg2e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 22 <unknown>

vsseg2e32.v v24, (a0)
# CHECK-INST: vsseg2e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 22 <unknown>

vsseg2e64.v v24, (a0)
# CHECK-INST: vsseg2e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 22 <unknown>

vsseg2e128.v v24, (a0)
# CHECK-INST: vsseg2e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 32 <unknown>

vsseg2e256.v v24, (a0)
# CHECK-INST: vsseg2e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 32 <unknown>

vsseg2e512.v v24, (a0)
# CHECK-INST: vsseg2e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 32 <unknown>

vsseg2e1024.v v24, (a0)
# CHECK-INST: vsseg2e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 32 <unknown>

vsseg2e8.v v24, (a0), v0.t
# CHECK-INST: vsseg2e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 20 <unknown>

vsseg2e16.v v24, (a0), v0.t
# CHECK-INST: vsseg2e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 20 <unknown>

vsseg2e32.v v24, (a0), v0.t
# CHECK-INST: vsseg2e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 20 <unknown>

vsseg2e64.v v24, (a0), v0.t
# CHECK-INST: vsseg2e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 20 <unknown>

vsseg2e128.v v24, (a0), v0.t
# CHECK-INST: vsseg2e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 30 <unknown>

vsseg2e256.v v24, (a0), v0.t
# CHECK-INST: vsseg2e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 30 <unknown>

vsseg2e512.v v24, (a0), v0.t
# CHECK-INST: vsseg2e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 30 <unknown>

vsseg2e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg2e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 30 <unknown>

vsseg3e8.v v24, (a0)
# CHECK-INST: vsseg3e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 42 <unknown>

vsseg3e16.v v24, (a0)
# CHECK-INST: vsseg3e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 42 <unknown>

vsseg3e32.v v24, (a0)
# CHECK-INST: vsseg3e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 42 <unknown>

vsseg3e64.v v24, (a0)
# CHECK-INST: vsseg3e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 42 <unknown>

vsseg3e128.v v24, (a0)
# CHECK-INST: vsseg3e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 52 <unknown>

vsseg3e256.v v24, (a0)
# CHECK-INST: vsseg3e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 52 <unknown>

vsseg3e512.v v24, (a0)
# CHECK-INST: vsseg3e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 52 <unknown>

vsseg3e1024.v v24, (a0)
# CHECK-INST: vsseg3e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 52 <unknown>

vsseg3e8.v v24, (a0), v0.t
# CHECK-INST: vsseg3e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 40 <unknown>

vsseg3e16.v v24, (a0), v0.t
# CHECK-INST: vsseg3e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 40 <unknown>

vsseg3e32.v v24, (a0), v0.t
# CHECK-INST: vsseg3e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 40 <unknown>

vsseg3e64.v v24, (a0), v0.t
# CHECK-INST: vsseg3e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 40 <unknown>

vsseg3e128.v v24, (a0), v0.t
# CHECK-INST: vsseg3e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 50 <unknown>

vsseg3e256.v v24, (a0), v0.t
# CHECK-INST: vsseg3e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 50 <unknown>

vsseg3e512.v v24, (a0), v0.t
# CHECK-INST: vsseg3e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 50 <unknown>

vsseg3e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg3e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 50 <unknown>

vsseg4e8.v v24, (a0)
# CHECK-INST: vsseg4e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 62 <unknown>

vsseg4e16.v v24, (a0)
# CHECK-INST: vsseg4e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 62 <unknown>

vsseg4e32.v v24, (a0)
# CHECK-INST: vsseg4e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 62 <unknown>

vsseg4e64.v v24, (a0)
# CHECK-INST: vsseg4e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 62 <unknown>

vsseg4e128.v v24, (a0)
# CHECK-INST: vsseg4e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 72 <unknown>

vsseg4e256.v v24, (a0)
# CHECK-INST: vsseg4e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 72 <unknown>

vsseg4e512.v v24, (a0)
# CHECK-INST: vsseg4e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 72 <unknown>

vsseg4e1024.v v24, (a0)
# CHECK-INST: vsseg4e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 72 <unknown>

vsseg4e8.v v24, (a0), v0.t
# CHECK-INST: vsseg4e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 60 <unknown>

vsseg4e16.v v24, (a0), v0.t
# CHECK-INST: vsseg4e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 60 <unknown>

vsseg4e32.v v24, (a0), v0.t
# CHECK-INST: vsseg4e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 60 <unknown>

vsseg4e64.v v24, (a0), v0.t
# CHECK-INST: vsseg4e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 60 <unknown>

vsseg4e128.v v24, (a0), v0.t
# CHECK-INST: vsseg4e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 70 <unknown>

vsseg4e256.v v24, (a0), v0.t
# CHECK-INST: vsseg4e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 70 <unknown>

vsseg4e512.v v24, (a0), v0.t
# CHECK-INST: vsseg4e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 70 <unknown>

vsseg4e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg4e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 70 <unknown>

vsseg5e8.v v24, (a0)
# CHECK-INST: vsseg5e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 82 <unknown>

vsseg5e16.v v24, (a0)
# CHECK-INST: vsseg5e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 82 <unknown>

vsseg5e32.v v24, (a0)
# CHECK-INST: vsseg5e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 82 <unknown>

vsseg5e64.v v24, (a0)
# CHECK-INST: vsseg5e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 82 <unknown>

vsseg5e128.v v24, (a0)
# CHECK-INST: vsseg5e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 92 <unknown>

vsseg5e256.v v24, (a0)
# CHECK-INST: vsseg5e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 92 <unknown>

vsseg5e512.v v24, (a0)
# CHECK-INST: vsseg5e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 92 <unknown>

vsseg5e1024.v v24, (a0)
# CHECK-INST: vsseg5e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 92 <unknown>

vsseg5e8.v v24, (a0), v0.t
# CHECK-INST: vsseg5e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 80 <unknown>

vsseg5e16.v v24, (a0), v0.t
# CHECK-INST: vsseg5e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 80 <unknown>

vsseg5e32.v v24, (a0), v0.t
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 80 <unknown>

vsseg5e64.v v24, (a0), v0.t
# CHECK-INST: vsseg5e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 80 <unknown>

vsseg5e128.v v24, (a0), v0.t
# CHECK-INST: vsseg5e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 90 <unknown>

vsseg5e256.v v24, (a0), v0.t
# CHECK-INST: vsseg5e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 90 <unknown>

vsseg5e512.v v24, (a0), v0.t
# CHECK-INST: vsseg5e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 90 <unknown>

vsseg5e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg5e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 90 <unknown>

vsseg6e8.v v24, (a0)
# CHECK-INST: vsseg6e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 a2 <unknown>

vsseg6e16.v v24, (a0)
# CHECK-INST: vsseg6e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 a2 <unknown>

vsseg6e32.v v24, (a0)
# CHECK-INST: vsseg6e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 a2 <unknown>

vsseg6e64.v v24, (a0)
# CHECK-INST: vsseg6e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 a2 <unknown>

vsseg6e128.v v24, (a0)
# CHECK-INST: vsseg6e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 b2 <unknown>

vsseg6e256.v v24, (a0)
# CHECK-INST: vsseg6e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 b2 <unknown>

vsseg6e512.v v24, (a0)
# CHECK-INST: vsseg6e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 b2 <unknown>

vsseg6e1024.v v24, (a0)
# CHECK-INST: vsseg6e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 b2 <unknown>

vsseg6e8.v v24, (a0), v0.t
# CHECK-INST: vsseg6e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 a0 <unknown>

vsseg6e16.v v24, (a0), v0.t
# CHECK-INST: vsseg6e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 a0 <unknown>

vsseg6e32.v v24, (a0), v0.t
# CHECK-INST: vsseg6e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 a0 <unknown>

vsseg6e64.v v24, (a0), v0.t
# CHECK-INST: vsseg6e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 a0 <unknown>

vsseg6e128.v v24, (a0), v0.t
# CHECK-INST: vsseg6e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 b0 <unknown>

vsseg6e256.v v24, (a0), v0.t
# CHECK-INST: vsseg6e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 b0 <unknown>

vsseg6e512.v v24, (a0), v0.t
# CHECK-INST: vsseg6e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 b0 <unknown>

vsseg6e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg6e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 b0 <unknown>

vsseg7e8.v v24, (a0)
# CHECK-INST: vsseg7e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 c2 <unknown>

vsseg7e16.v v24, (a0)
# CHECK-INST: vsseg7e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 c2 <unknown>

vsseg7e32.v v24, (a0)
# CHECK-INST: vsseg7e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 c2 <unknown>

vsseg7e64.v v24, (a0)
# CHECK-INST: vsseg7e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 c2 <unknown>

vsseg7e128.v v24, (a0)
# CHECK-INST: vsseg7e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 d2 <unknown>

vsseg7e256.v v24, (a0)
# CHECK-INST: vsseg7e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 d2 <unknown>

vsseg7e512.v v24, (a0)
# CHECK-INST: vsseg7e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 d2 <unknown>

vsseg7e1024.v v24, (a0)
# CHECK-INST: vsseg7e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 d2 <unknown>

vsseg7e8.v v24, (a0), v0.t
# CHECK-INST: vsseg7e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 c0 <unknown>

vsseg7e16.v v24, (a0), v0.t
# CHECK-INST: vsseg7e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 c0 <unknown>

vsseg7e32.v v24, (a0), v0.t
# CHECK-INST: vsseg7e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 c0 <unknown>

vsseg7e64.v v24, (a0), v0.t
# CHECK-INST: vsseg7e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 c0 <unknown>

vsseg7e128.v v24, (a0), v0.t
# CHECK-INST: vsseg7e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 d0 <unknown>

vsseg7e256.v v24, (a0), v0.t
# CHECK-INST: vsseg7e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 d0 <unknown>

vsseg7e512.v v24, (a0), v0.t
# CHECK-INST: vsseg7e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 d0 <unknown>

vsseg7e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg7e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 d0 <unknown>

vsseg8e8.v v24, (a0)
# CHECK-INST: vsseg8e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 e2 <unknown>

vsseg8e16.v v24, (a0)
# CHECK-INST: vsseg8e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 e2 <unknown>

vsseg8e32.v v24, (a0)
# CHECK-INST: vsseg8e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 e2 <unknown>

vsseg8e64.v v24, (a0)
# CHECK-INST: vsseg8e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 e2 <unknown>

vsseg8e128.v v24, (a0)
# CHECK-INST: vsseg8e128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 f2 <unknown>

vsseg8e256.v v24, (a0)
# CHECK-INST: vsseg8e256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 f2 <unknown>

vsseg8e512.v v24, (a0)
# CHECK-INST: vsseg8e512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 f2 <unknown>

vsseg8e1024.v v24, (a0)
# CHECK-INST: vsseg8e1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 f2 <unknown>

vsseg8e8.v v24, (a0), v0.t
# CHECK-INST: vsseg8e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 e0 <unknown>

vsseg8e16.v v24, (a0), v0.t
# CHECK-INST: vsseg8e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 e0 <unknown>

vsseg8e32.v v24, (a0), v0.t
# CHECK-INST: vsseg8e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 e0 <unknown>

vsseg8e64.v v24, (a0), v0.t
# CHECK-INST: vsseg8e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 e0 <unknown>

vsseg8e128.v v24, (a0), v0.t
# CHECK-INST: vsseg8e128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 05 f0 <unknown>

vsseg8e256.v v24, (a0), v0.t
# CHECK-INST: vsseg8e256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 05 f0 <unknown>

vsseg8e512.v v24, (a0), v0.t
# CHECK-INST: vsseg8e512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 05 f0 <unknown>

vsseg8e1024.v v24, (a0), v0.t
# CHECK-INST: vsseg8e1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 05 f0 <unknown>

vssseg2e8.v v24, (a0), a1
# CHECK-INST: vssseg2e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 2a <unknown>

vssseg2e16.v v24, (a0), a1
# CHECK-INST: vssseg2e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 2a <unknown>

vssseg2e32.v v24, (a0), a1
# CHECK-INST: vssseg2e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 2a <unknown>

vssseg2e64.v v24, (a0), a1
# CHECK-INST: vssseg2e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 2a <unknown>

vssseg2e128.v v24, (a0), a1
# CHECK-INST: vssseg2e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 3a <unknown>

vssseg2e256.v v24, (a0), a1
# CHECK-INST: vssseg2e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 3a <unknown>

vssseg2e512.v v24, (a0), a1
# CHECK-INST: vssseg2e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 3a <unknown>

vssseg2e1024.v v24, (a0), a1
# CHECK-INST: vssseg2e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 3a <unknown>

vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 28 <unknown>

vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 28 <unknown>

vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 28 <unknown>

vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 28 <unknown>

vssseg2e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 38 <unknown>

vssseg2e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 38 <unknown>

vssseg2e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 38 <unknown>

vssseg2e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 38 <unknown>

vssseg3e8.v v24, (a0), a1
# CHECK-INST: vssseg3e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 4a <unknown>

vssseg3e16.v v24, (a0), a1
# CHECK-INST: vssseg3e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 4a <unknown>

vssseg3e32.v v24, (a0), a1
# CHECK-INST: vssseg3e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 4a <unknown>

vssseg3e64.v v24, (a0), a1
# CHECK-INST: vssseg3e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 4a <unknown>

vssseg3e128.v v24, (a0), a1
# CHECK-INST: vssseg3e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 5a <unknown>

vssseg3e256.v v24, (a0), a1
# CHECK-INST: vssseg3e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 5a <unknown>

vssseg3e512.v v24, (a0), a1
# CHECK-INST: vssseg3e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 5a <unknown>

vssseg3e1024.v v24, (a0), a1
# CHECK-INST: vssseg3e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 5a <unknown>

vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 48 <unknown>

vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 48 <unknown>

vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 48 <unknown>

vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 48 <unknown>

vssseg3e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 58 <unknown>

vssseg3e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 58 <unknown>

vssseg3e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 58 <unknown>

vssseg3e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 58 <unknown>

vssseg4e8.v v24, (a0), a1
# CHECK-INST: vssseg4e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 6a <unknown>

vssseg4e16.v v24, (a0), a1
# CHECK-INST: vssseg4e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 6a <unknown>

vssseg4e32.v v24, (a0), a1
# CHECK-INST: vssseg4e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 6a <unknown>

vssseg4e64.v v24, (a0), a1
# CHECK-INST: vssseg4e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 6a <unknown>

vssseg4e128.v v24, (a0), a1
# CHECK-INST: vssseg4e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 7a <unknown>

vssseg4e256.v v24, (a0), a1
# CHECK-INST: vssseg4e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 7a <unknown>

vssseg4e512.v v24, (a0), a1
# CHECK-INST: vssseg4e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 7a <unknown>

vssseg4e1024.v v24, (a0), a1
# CHECK-INST: vssseg4e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 7a <unknown>

vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 68 <unknown>

vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 68 <unknown>

vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 68 <unknown>

vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 68 <unknown>

vssseg4e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 78 <unknown>

vssseg4e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 78 <unknown>

vssseg4e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 78 <unknown>

vssseg4e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 78 <unknown>

vssseg5e8.v v24, (a0), a1
# CHECK-INST: vssseg5e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 8a <unknown>

vssseg5e16.v v24, (a0), a1
# CHECK-INST: vssseg5e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 8a <unknown>

vssseg5e32.v v24, (a0), a1
# CHECK-INST: vssseg5e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 8a <unknown>

vssseg5e64.v v24, (a0), a1
# CHECK-INST: vssseg5e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 8a <unknown>

vssseg5e128.v v24, (a0), a1
# CHECK-INST: vssseg5e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 9a <unknown>

vssseg5e256.v v24, (a0), a1
# CHECK-INST: vssseg5e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 9a <unknown>

vssseg5e512.v v24, (a0), a1
# CHECK-INST: vssseg5e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 9a <unknown>

vssseg5e1024.v v24, (a0), a1
# CHECK-INST: vssseg5e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 9a <unknown>

vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 88 <unknown>

vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 88 <unknown>

vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 88 <unknown>

vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 88 <unknown>

vssseg5e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 98 <unknown>

vssseg5e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 98 <unknown>

vssseg5e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 98 <unknown>

vssseg5e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 98 <unknown>

vssseg6e8.v v24, (a0), a1
# CHECK-INST: vssseg6e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 aa <unknown>

vssseg6e16.v v24, (a0), a1
# CHECK-INST: vssseg6e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 aa <unknown>

vssseg6e32.v v24, (a0), a1
# CHECK-INST: vssseg6e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 aa <unknown>

vssseg6e64.v v24, (a0), a1
# CHECK-INST: vssseg6e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 aa <unknown>

vssseg6e128.v v24, (a0), a1
# CHECK-INST: vssseg6e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 ba <unknown>

vssseg6e256.v v24, (a0), a1
# CHECK-INST: vssseg6e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 ba <unknown>

vssseg6e512.v v24, (a0), a1
# CHECK-INST: vssseg6e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 ba <unknown>

vssseg6e1024.v v24, (a0), a1
# CHECK-INST: vssseg6e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 ba <unknown>

vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 a8 <unknown>

vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 a8 <unknown>

vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 a8 <unknown>

vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 a8 <unknown>

vssseg6e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 b8 <unknown>

vssseg6e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 b8 <unknown>

vssseg6e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 b8 <unknown>

vssseg6e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 b8 <unknown>

vssseg7e8.v v24, (a0), a1
# CHECK-INST: vssseg7e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 ca <unknown>

vssseg7e16.v v24, (a0), a1
# CHECK-INST: vssseg7e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 ca <unknown>

vssseg7e32.v v24, (a0), a1
# CHECK-INST: vssseg7e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 ca <unknown>

vssseg7e64.v v24, (a0), a1
# CHECK-INST: vssseg7e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 ca <unknown>

vssseg7e128.v v24, (a0), a1
# CHECK-INST: vssseg7e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 da <unknown>

vssseg7e256.v v24, (a0), a1
# CHECK-INST: vssseg7e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 da <unknown>

vssseg7e512.v v24, (a0), a1
# CHECK-INST: vssseg7e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 da <unknown>

vssseg7e1024.v v24, (a0), a1
# CHECK-INST: vssseg7e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 da <unknown>

vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 c8 <unknown>

vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 c8 <unknown>

vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 c8 <unknown>

vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 c8 <unknown>

vssseg7e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 d8 <unknown>

vssseg7e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 d8 <unknown>

vssseg7e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 d8 <unknown>

vssseg7e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 d8 <unknown>

vssseg8e8.v v24, (a0), a1
# CHECK-INST: vssseg8e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 ea <unknown>

vssseg8e16.v v24, (a0), a1
# CHECK-INST: vssseg8e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 ea <unknown>

vssseg8e32.v v24, (a0), a1
# CHECK-INST: vssseg8e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 ea <unknown>

vssseg8e64.v v24, (a0), a1
# CHECK-INST: vssseg8e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 ea <unknown>

vssseg8e128.v v24, (a0), a1
# CHECK-INST: vssseg8e128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 fa <unknown>

vssseg8e256.v v24, (a0), a1
# CHECK-INST: vssseg8e256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 fa <unknown>

vssseg8e512.v v24, (a0), a1
# CHECK-INST: vssseg8e512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 fa <unknown>

vssseg8e1024.v v24, (a0), a1
# CHECK-INST: vssseg8e1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 fa <unknown>

vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 e8 <unknown>

vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 e8 <unknown>

vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 e8 <unknown>

vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 e8 <unknown>

vssseg8e128.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c b5 f8 <unknown>

vssseg8e256.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c b5 f8 <unknown>

vssseg8e512.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c b5 f8 <unknown>

vssseg8e1024.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c b5 f8 <unknown>

vsxseg2ei8.v v24, (a0), v4
# CHECK-INST: vsxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 2e <unknown>

vsxseg2ei16.v v24, (a0), v4
# CHECK-INST: vsxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 2e <unknown>

vsxseg2ei32.v v24, (a0), v4
# CHECK-INST: vsxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 2e <unknown>

vsxseg2ei64.v v24, (a0), v4
# CHECK-INST: vsxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 2e <unknown>

vsxseg2ei128.v v24, (a0), v4
# CHECK-INST: vsxseg2ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 3e <unknown>

vsxseg2ei256.v v24, (a0), v4
# CHECK-INST: vsxseg2ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 3e <unknown>

vsxseg2ei512.v v24, (a0), v4
# CHECK-INST: vsxseg2ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 3e <unknown>

vsxseg2ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg2ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 3e <unknown>

vsxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 2c <unknown>

vsxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 2c <unknown>

vsxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 2c <unknown>

vsxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 2c <unknown>

vsxseg2ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 3c <unknown>

vsxseg2ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 3c <unknown>

vsxseg2ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 3c <unknown>

vsxseg2ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg2ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 3c <unknown>

vsxseg3ei8.v v24, (a0), v4
# CHECK-INST: vsxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 4e <unknown>

vsxseg3ei16.v v24, (a0), v4
# CHECK-INST: vsxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 4e <unknown>

vsxseg3ei32.v v24, (a0), v4
# CHECK-INST: vsxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 4e <unknown>

vsxseg3ei64.v v24, (a0), v4
# CHECK-INST: vsxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 4e <unknown>

vsxseg3ei128.v v24, (a0), v4
# CHECK-INST: vsxseg3ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 5e <unknown>

vsxseg3ei256.v v24, (a0), v4
# CHECK-INST: vsxseg3ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 5e <unknown>

vsxseg3ei512.v v24, (a0), v4
# CHECK-INST: vsxseg3ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 5e <unknown>

vsxseg3ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg3ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 5e <unknown>

vsxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 4c <unknown>

vsxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 4c <unknown>

vsxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 4c <unknown>

vsxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 4c <unknown>

vsxseg3ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 5c <unknown>

vsxseg3ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 5c <unknown>

vsxseg3ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 5c <unknown>

vsxseg3ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg3ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 5c <unknown>

vsxseg4ei8.v v24, (a0), v4
# CHECK-INST: vsxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 6e <unknown>

vsxseg4ei16.v v24, (a0), v4
# CHECK-INST: vsxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 6e <unknown>

vsxseg4ei32.v v24, (a0), v4
# CHECK-INST: vsxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 6e <unknown>

vsxseg4ei64.v v24, (a0), v4
# CHECK-INST: vsxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 6e <unknown>

vsxseg4ei128.v v24, (a0), v4
# CHECK-INST: vsxseg4ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 7e <unknown>

vsxseg4ei256.v v24, (a0), v4
# CHECK-INST: vsxseg4ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 7e <unknown>

vsxseg4ei512.v v24, (a0), v4
# CHECK-INST: vsxseg4ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 7e <unknown>

vsxseg4ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg4ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 7e <unknown>

vsxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 6c <unknown>

vsxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 6c <unknown>

vsxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 6c <unknown>

vsxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 6c <unknown>

vsxseg4ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 7c <unknown>

vsxseg4ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 7c <unknown>

vsxseg4ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 7c <unknown>

vsxseg4ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg4ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 7c <unknown>

vsxseg5ei8.v v24, (a0), v4
# CHECK-INST: vsxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 8e <unknown>

vsxseg5ei16.v v24, (a0), v4
# CHECK-INST: vsxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 8e <unknown>

vsxseg5ei32.v v24, (a0), v4
# CHECK-INST: vsxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 8e <unknown>

vsxseg5ei64.v v24, (a0), v4
# CHECK-INST: vsxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 8e <unknown>

vsxseg5ei128.v v24, (a0), v4
# CHECK-INST: vsxseg5ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 9e <unknown>

vsxseg5ei256.v v24, (a0), v4
# CHECK-INST: vsxseg5ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 9e <unknown>

vsxseg5ei512.v v24, (a0), v4
# CHECK-INST: vsxseg5ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 9e <unknown>

vsxseg5ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg5ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 9e <unknown>

vsxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 8c <unknown>

vsxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 8c <unknown>

vsxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 8c <unknown>

vsxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 8c <unknown>

vsxseg5ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 9c <unknown>

vsxseg5ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 9c <unknown>

vsxseg5ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 9c <unknown>

vsxseg5ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg5ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 9c <unknown>

vsxseg6ei8.v v24, (a0), v4
# CHECK-INST: vsxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 ae <unknown>

vsxseg6ei16.v v24, (a0), v4
# CHECK-INST: vsxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 ae <unknown>

vsxseg6ei32.v v24, (a0), v4
# CHECK-INST: vsxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 ae <unknown>

vsxseg6ei64.v v24, (a0), v4
# CHECK-INST: vsxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 ae <unknown>

vsxseg6ei128.v v24, (a0), v4
# CHECK-INST: vsxseg6ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 be <unknown>

vsxseg6ei256.v v24, (a0), v4
# CHECK-INST: vsxseg6ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 be <unknown>

vsxseg6ei512.v v24, (a0), v4
# CHECK-INST: vsxseg6ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 be <unknown>

vsxseg6ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg6ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 be <unknown>

vsxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 ac <unknown>

vsxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 ac <unknown>

vsxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 ac <unknown>

vsxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 ac <unknown>

vsxseg6ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 bc <unknown>

vsxseg6ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 bc <unknown>

vsxseg6ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 bc <unknown>

vsxseg6ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg6ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 bc <unknown>

vsxseg7ei8.v v24, (a0), v4
# CHECK-INST: vsxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 ce <unknown>

vsxseg7ei16.v v24, (a0), v4
# CHECK-INST: vsxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 ce <unknown>

vsxseg7ei32.v v24, (a0), v4
# CHECK-INST: vsxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 ce <unknown>

vsxseg7ei64.v v24, (a0), v4
# CHECK-INST: vsxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 ce <unknown>

vsxseg7ei128.v v24, (a0), v4
# CHECK-INST: vsxseg7ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 de <unknown>

vsxseg7ei256.v v24, (a0), v4
# CHECK-INST: vsxseg7ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 de <unknown>

vsxseg7ei512.v v24, (a0), v4
# CHECK-INST: vsxseg7ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 de <unknown>

vsxseg7ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg7ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 de <unknown>

vsxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 cc <unknown>

vsxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 cc <unknown>

vsxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 cc <unknown>

vsxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 cc <unknown>

vsxseg7ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 dc <unknown>

vsxseg7ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 dc <unknown>

vsxseg7ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 dc <unknown>

vsxseg7ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg7ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 dc <unknown>

vsxseg8ei8.v v24, (a0), v4
# CHECK-INST: vsxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 ee <unknown>

vsxseg8ei16.v v24, (a0), v4
# CHECK-INST: vsxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 ee <unknown>

vsxseg8ei32.v v24, (a0), v4
# CHECK-INST: vsxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 ee <unknown>

vsxseg8ei64.v v24, (a0), v4
# CHECK-INST: vsxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 ee <unknown>

vsxseg8ei128.v v24, (a0), v4
# CHECK-INST: vsxseg8ei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 fe <unknown>

vsxseg8ei256.v v24, (a0), v4
# CHECK-INST: vsxseg8ei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 fe <unknown>

vsxseg8ei512.v v24, (a0), v4
# CHECK-INST: vsxseg8ei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 fe <unknown>

vsxseg8ei1024.v v24, (a0), v4
# CHECK-INST: vsxseg8ei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 fe <unknown>

vsxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 ec <unknown>

vsxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 ec <unknown>

vsxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 ec <unknown>

vsxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 ec <unknown>

vsxseg8ei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 0c 45 fc <unknown>

vsxseg8ei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 5c 45 fc <unknown>

vsxseg8ei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 6c 45 fc <unknown>

vsxseg8ei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxseg8ei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg' (Vector segment load/store instructions)
# CHECK-UNKNOWN: 27 7c 45 fc <unknown>