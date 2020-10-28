# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+a,+experimental-zvamo %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+a,+experimental-zvamo %s \
# RUN:        | llvm-objdump -d --mattr=+a,+experimental-zvamo - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+a,+experimental-zvamo %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


vamoswapei8.v v8, (a0), v4, v8
# CHECK-INST: vamoswapei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 0e <unknown>

vamoswapei16.v v8, (a0), v4, v8
# CHECK-INST: vamoswapei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 0e <unknown>

vamoswapei32.v v8, (a0), v4, v8
# CHECK-INST: vamoswapei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 0e <unknown>

vamoswapei64.v v8, (a0), v4, v8
# CHECK-INST: vamoswapei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 0e <unknown>

vamoswapei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoswapei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 0c <unknown>

vamoswapei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoswapei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 0c <unknown>

vamoswapei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoswapei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 0c <unknown>

vamoswapei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoswapei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 0c <unknown>

vamoaddei8.v v8, (a0), v4, v8
# CHECK-INST: vamoaddei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 06 <unknown>

vamoaddei16.v v8, (a0), v4, v8
# CHECK-INST: vamoaddei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 06 <unknown>

vamoaddei32.v v8, (a0), v4, v8
# CHECK-INST: vamoaddei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 06 <unknown>

vamoaddei64.v v8, (a0), v4, v8
# CHECK-INST: vamoaddei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 06 <unknown>

vamoaddei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoaddei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 04 <unknown>

vamoaddei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoaddei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 04 <unknown>

vamoaddei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoaddei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 04 <unknown>

vamoaddei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoaddei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 04 <unknown>

vamoxorei8.v v8, (a0), v4, v8
# CHECK-INST: vamoxorei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 26 <unknown>

vamoxorei16.v v8, (a0), v4, v8
# CHECK-INST: vamoxorei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 26 <unknown>

vamoxorei32.v v8, (a0), v4, v8
# CHECK-INST: vamoxorei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 26 <unknown>

vamoxorei64.v v8, (a0), v4, v8
# CHECK-INST: vamoxorei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 26 <unknown>

vamoxorei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoxorei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 24 <unknown>

vamoxorei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoxorei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 24 <unknown>

vamoxorei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoxorei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 24 <unknown>

vamoxorei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoxorei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 24 <unknown>

vamoandei8.v v8, (a0), v4, v8
# CHECK-INST: vamoandei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 66 <unknown>

vamoandei16.v v8, (a0), v4, v8
# CHECK-INST: vamoandei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 66 <unknown>

vamoandei32.v v8, (a0), v4, v8
# CHECK-INST: vamoandei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 66 <unknown>

vamoandei64.v v8, (a0), v4, v8
# CHECK-INST: vamoandei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 66 <unknown>

vamoandei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoandei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 64 <unknown>

vamoandei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoandei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 64 <unknown>

vamoandei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoandei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 64 <unknown>

vamoandei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoandei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 64 <unknown>

vamoorei8.v v8, (a0), v4, v8
# CHECK-INST: vamoorei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 46 <unknown>

vamoorei16.v v8, (a0), v4, v8
# CHECK-INST: vamoorei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 46 <unknown>

vamoorei32.v v8, (a0), v4, v8
# CHECK-INST: vamoorei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 46 <unknown>

vamoorei64.v v8, (a0), v4, v8
# CHECK-INST: vamoorei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 46 <unknown>

vamoorei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoorei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 44 <unknown>

vamoorei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoorei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 44 <unknown>

vamoorei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoorei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 44 <unknown>

vamoorei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamoorei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 44 <unknown>

vamominei8.v v8, (a0), v4, v8
# CHECK-INST: vamominei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 86 <unknown>

vamominei16.v v8, (a0), v4, v8
# CHECK-INST: vamominei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 86 <unknown>

vamominei32.v v8, (a0), v4, v8
# CHECK-INST: vamominei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 86 <unknown>

vamominei64.v v8, (a0), v4, v8
# CHECK-INST: vamominei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 86 <unknown>

vamominei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 84 <unknown>

vamominei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 84 <unknown>

vamominei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 84 <unknown>

vamominei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 84 <unknown>

vamomaxei8.v v8, (a0), v4, v8
# CHECK-INST: vamomaxei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 a6 <unknown>

vamomaxei16.v v8, (a0), v4, v8
# CHECK-INST: vamomaxei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 a6 <unknown>

vamomaxei32.v v8, (a0), v4, v8
# CHECK-INST: vamomaxei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 a6 <unknown>

vamomaxei64.v v8, (a0), v4, v8
# CHECK-INST: vamomaxei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 a6 <unknown>

vamomaxei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 a4 <unknown>

vamomaxei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 a4 <unknown>

vamomaxei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 a4 <unknown>

vamomaxei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 a4 <unknown>

vamominuei8.v v8, (a0), v4, v8
# CHECK-INST: vamominuei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 c6 <unknown>

vamominuei16.v v8, (a0), v4, v8
# CHECK-INST: vamominuei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 c6 <unknown>

vamominuei32.v v8, (a0), v4, v8
# CHECK-INST: vamominuei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 c6 <unknown>

vamominuei64.v v8, (a0), v4, v8
# CHECK-INST: vamominuei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 c6 <unknown>

vamominuei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominuei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 c4 <unknown>

vamominuei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominuei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 c4 <unknown>

vamominuei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominuei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 c4 <unknown>

vamominuei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamominuei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 c4 <unknown>

vamomaxuei8.v v8, (a0), v4, v8
# CHECK-INST: vamomaxuei8.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x04,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 e6 <unknown>

vamomaxuei16.v v8, (a0), v4, v8
# CHECK-INST: vamomaxuei16.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x54,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 e6 <unknown>

vamomaxuei32.v v8, (a0), v4, v8
# CHECK-INST: vamomaxuei32.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x64,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 e6 <unknown>

vamomaxuei64.v v8, (a0), v4, v8
# CHECK-INST: vamomaxuei64.v v8, (a0), v4, v8
# CHECK-ENCODING: [0x2f,0x74,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 e6 <unknown>

vamomaxuei8.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxuei8.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x04,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 04 45 e4 <unknown>

vamomaxuei16.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxuei16.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x54,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 54 45 e4 <unknown>

vamomaxuei32.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxuei32.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x64,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 64 45 e4 <unknown>

vamomaxuei64.v v8, (a0), v4, v8, v0.t
# CHECK-INST: vamomaxuei64.v v8, (a0), v4, v8, v0.t
# CHECK-ENCODING: [0x2f,0x74,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 74 45 e4 <unknown>

vamoswapei8.v x0, (a0), v4, v24
# CHECK-INST: vamoswapei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 0a <unknown>

vamoswapei16.v x0, (a0), v4, v24
# CHECK-INST: vamoswapei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 0a <unknown>

vamoswapei32.v x0, (a0), v4, v24
# CHECK-INST: vamoswapei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 0a <unknown>

vamoswapei64.v x0, (a0), v4, v24
# CHECK-INST: vamoswapei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 0a <unknown>

vamoswapei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoswapei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 08 <unknown>

vamoswapei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoswapei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 08 <unknown>

vamoswapei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoswapei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 08 <unknown>

vamoswapei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoswapei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 08 <unknown>

vamoaddei8.v x0, (a0), v4, v24
# CHECK-INST: vamoaddei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 02 <unknown>

vamoaddei16.v x0, (a0), v4, v24
# CHECK-INST: vamoaddei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 02 <unknown>

vamoaddei32.v x0, (a0), v4, v24
# CHECK-INST: vamoaddei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 02 <unknown>

vamoaddei64.v x0, (a0), v4, v24
# CHECK-INST: vamoaddei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 02 <unknown>

vamoaddei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoaddei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 00 <unknown>

vamoaddei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoaddei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 00 <unknown>

vamoaddei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoaddei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 00 <unknown>

vamoaddei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoaddei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 00 <unknown>

vamoxorei8.v x0, (a0), v4, v24
# CHECK-INST: vamoxorei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 22 <unknown>

vamoxorei16.v x0, (a0), v4, v24
# CHECK-INST: vamoxorei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 22 <unknown>

vamoxorei32.v x0, (a0), v4, v24
# CHECK-INST: vamoxorei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 22 <unknown>

vamoxorei64.v x0, (a0), v4, v24
# CHECK-INST: vamoxorei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 22 <unknown>

vamoxorei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoxorei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 20 <unknown>

vamoxorei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoxorei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 20 <unknown>

vamoxorei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoxorei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 20 <unknown>

vamoxorei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoxorei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 20 <unknown>

vamoandei8.v x0, (a0), v4, v24
# CHECK-INST: vamoandei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 62 <unknown>

vamoandei16.v x0, (a0), v4, v24
# CHECK-INST: vamoandei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 62 <unknown>

vamoandei32.v x0, (a0), v4, v24
# CHECK-INST: vamoandei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 62 <unknown>

vamoandei64.v x0, (a0), v4, v24
# CHECK-INST: vamoandei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 62 <unknown>

vamoandei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoandei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 60 <unknown>

vamoandei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoandei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 60 <unknown>

vamoandei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoandei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 60 <unknown>

vamoandei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoandei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 60 <unknown>

vamoorei8.v x0, (a0), v4, v24
# CHECK-INST: vamoorei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x42]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 42 <unknown>

vamoorei16.v x0, (a0), v4, v24
# CHECK-INST: vamoorei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x42]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 42 <unknown>

vamoorei32.v x0, (a0), v4, v24
# CHECK-INST: vamoorei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x42]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 42 <unknown>

vamoorei64.v x0, (a0), v4, v24
# CHECK-INST: vamoorei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x42]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 42 <unknown>

vamoorei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoorei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x40]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 40 <unknown>

vamoorei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoorei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x40]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 40 <unknown>

vamoorei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoorei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x40]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 40 <unknown>

vamoorei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamoorei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x40]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 40 <unknown>

vamominei8.v x0, (a0), v4, v24
# CHECK-INST: vamominei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 82 <unknown>

vamominei16.v x0, (a0), v4, v24
# CHECK-INST: vamominei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 82 <unknown>

vamominei32.v x0, (a0), v4, v24
# CHECK-INST: vamominei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 82 <unknown>

vamominei64.v x0, (a0), v4, v24
# CHECK-INST: vamominei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 82 <unknown>

vamominei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 80 <unknown>

vamominei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 80 <unknown>

vamominei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 80 <unknown>

vamominei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 80 <unknown>

vamomaxei8.v x0, (a0), v4, v24
# CHECK-INST: vamomaxei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 a2 <unknown>

vamomaxei16.v x0, (a0), v4, v24
# CHECK-INST: vamomaxei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 a2 <unknown>

vamomaxei32.v x0, (a0), v4, v24
# CHECK-INST: vamomaxei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 a2 <unknown>

vamomaxei64.v x0, (a0), v4, v24
# CHECK-INST: vamomaxei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 a2 <unknown>

vamomaxei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 a0 <unknown>

vamomaxei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 a0 <unknown>

vamomaxei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 a0 <unknown>

vamomaxei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 a0 <unknown>

vamominuei8.v x0, (a0), v4, v24
# CHECK-INST: vamominuei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 c2 <unknown>

vamominuei16.v x0, (a0), v4, v24
# CHECK-INST: vamominuei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 c2 <unknown>

vamominuei32.v x0, (a0), v4, v24
# CHECK-INST: vamominuei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 c2 <unknown>

vamominuei64.v x0, (a0), v4, v24
# CHECK-INST: vamominuei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 c2 <unknown>

vamominuei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominuei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 c0 <unknown>

vamominuei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominuei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 c0 <unknown>

vamominuei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominuei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 c0 <unknown>

vamominuei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamominuei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 c0 <unknown>

vamomaxuei8.v x0, (a0), v4, v24
# CHECK-INST: vamomaxuei8.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 e2 <unknown>

vamomaxuei16.v x0, (a0), v4, v24
# CHECK-INST: vamomaxuei16.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 e2 <unknown>

vamomaxuei32.v x0, (a0), v4, v24
# CHECK-INST: vamomaxuei32.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 e2 <unknown>

vamomaxuei64.v x0, (a0), v4, v24
# CHECK-INST: vamomaxuei64.v x0, (a0), v4, v24
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 e2 <unknown>

vamomaxuei8.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxuei8.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x0c,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 0c 45 e0 <unknown>

vamomaxuei16.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxuei16.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x5c,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 5c 45 e0 <unknown>

vamomaxuei32.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxuei32.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x6c,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 6c 45 e0 <unknown>

vamomaxuei64.v x0, (a0), v4, v24, v0.t
# CHECK-INST: vamomaxuei64.v x0, (a0), v4, v24, v0.t
# CHECK-ENCODING: [0x2f,0x7c,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'A' (Atomic Instructions), 'Zvamo'(Vector AMO Operations)
# CHECK-UNKNOWN: 2f 7c 45 e0 <unknown>