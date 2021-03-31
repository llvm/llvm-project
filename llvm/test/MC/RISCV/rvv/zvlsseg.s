# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:   --mattr=+experimental-zvlsseg --riscv-no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v \
# RUN:   --mattr=+experimental-zvlsseg %s \
# RUN:   | llvm-objdump -d --mattr=+experimental-v --mattr=+experimental-zvlsseg \
# RUN:   --riscv-no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v \
# RUN:   --mattr=+experimental-zvlsseg %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vlseg2e8.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 20 <unknown>

vlseg2e8.v v8, (a0)
# CHECK-INST: vlseg2e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 22 <unknown>

vlseg2e16.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 20 <unknown>

vlseg2e16.v v8, (a0)
# CHECK-INST: vlseg2e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 22 <unknown>

vlseg2e32.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 20 <unknown>

vlseg2e32.v v8, (a0)
# CHECK-INST: vlseg2e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 22 <unknown>

vlseg2e64.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 20 <unknown>

vlseg2e64.v v8, (a0)
# CHECK-INST: vlseg2e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 22 <unknown>

vlseg2e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 21 <unknown>

vlseg2e8ff.v v8, (a0)
# CHECK-INST: vlseg2e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 23 <unknown>

vlseg2e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 21 <unknown>

vlseg2e16ff.v v8, (a0)
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 23 <unknown>

vlseg2e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 21 <unknown>

vlseg2e32ff.v v8, (a0)
# CHECK-INST: vlseg2e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 23 <unknown>

vlseg2e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 21 <unknown>

vlseg2e64ff.v v8, (a0)
# CHECK-INST: vlseg2e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 23 <unknown>

vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 28 <unknown>

vlsseg2e8.v v8, (a0), a1
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 2a <unknown>

vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 28 <unknown>

vlsseg2e16.v v8, (a0), a1
# CHECK-INST: vlsseg2e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 2a <unknown>

vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 28 <unknown>

vlsseg2e32.v v8, (a0), a1
# CHECK-INST: vlsseg2e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 2a <unknown>

vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 28 <unknown>

vlsseg2e64.v v8, (a0), a1
# CHECK-INST: vlsseg2e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 2a <unknown>

vluxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 24 <unknown>

vluxseg2ei8.v v8, (a0), v4
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 26 <unknown>

vluxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 24 <unknown>

vluxseg2ei16.v v8, (a0), v4
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 26 <unknown>

vluxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 24 <unknown>

vluxseg2ei32.v v8, (a0), v4
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 26 <unknown>

vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 24 <unknown>

vluxseg2ei64.v v8, (a0), v4
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 26 <unknown>

vloxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 2c <unknown>

vloxseg2ei8.v v8, (a0), v4
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 2e <unknown>

vloxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 2c <unknown>

vloxseg2ei16.v v8, (a0), v4
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 2e <unknown>

vloxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 2c <unknown>

vloxseg2ei32.v v8, (a0), v4
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 2e <unknown>

vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 2c <unknown>

vloxseg2ei64.v v8, (a0), v4
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 2e <unknown>

vlseg3e8.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 40 <unknown>

vlseg3e8.v v8, (a0)
# CHECK-INST: vlseg3e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 42 <unknown>

vlseg3e16.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 40 <unknown>

vlseg3e16.v v8, (a0)
# CHECK-INST: vlseg3e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 42 <unknown>

vlseg3e32.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 40 <unknown>

vlseg3e32.v v8, (a0)
# CHECK-INST: vlseg3e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 42 <unknown>

vlseg3e64.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 40 <unknown>

vlseg3e64.v v8, (a0)
# CHECK-INST: vlseg3e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 42 <unknown>

vlseg3e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 41 <unknown>

vlseg3e8ff.v v8, (a0)
# CHECK-INST: vlseg3e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 43 <unknown>

vlseg3e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 41 <unknown>

vlseg3e16ff.v v8, (a0)
# CHECK-INST: vlseg3e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 43 <unknown>

vlseg3e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 41 <unknown>

vlseg3e32ff.v v8, (a0)
# CHECK-INST: vlseg3e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 43 <unknown>

vlseg3e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 41 <unknown>

vlseg3e64ff.v v8, (a0)
# CHECK-INST: vlseg3e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 43 <unknown>

vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 48 <unknown>

vlsseg3e8.v v8, (a0), a1
# CHECK-INST: vlsseg3e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 4a <unknown>

vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 48 <unknown>

vlsseg3e16.v v8, (a0), a1
# CHECK-INST: vlsseg3e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 4a <unknown>

vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 48 <unknown>

vlsseg3e32.v v8, (a0), a1
# CHECK-INST: vlsseg3e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 4a <unknown>

vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 48 <unknown>

vlsseg3e64.v v8, (a0), a1
# CHECK-INST: vlsseg3e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 4a <unknown>

vluxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 44 <unknown>

vluxseg3ei8.v v8, (a0), v4
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 46 <unknown>

vluxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 44 <unknown>

vluxseg3ei16.v v8, (a0), v4
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 46 <unknown>

vluxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 44 <unknown>

vluxseg3ei32.v v8, (a0), v4
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 46 <unknown>

vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 44 <unknown>

vluxseg3ei64.v v8, (a0), v4
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 46 <unknown>

vloxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 4c <unknown>

vloxseg3ei8.v v8, (a0), v4
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 4e <unknown>

vloxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 4c <unknown>

vloxseg3ei16.v v8, (a0), v4
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 4e <unknown>

vloxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 4c <unknown>

vloxseg3ei32.v v8, (a0), v4
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 4e <unknown>

vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 4c <unknown>

vloxseg3ei64.v v8, (a0), v4
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 4e <unknown>

vlseg4e8.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 60 <unknown>

vlseg4e8.v v8, (a0)
# CHECK-INST: vlseg4e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 62 <unknown>

vlseg4e16.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 60 <unknown>

vlseg4e16.v v8, (a0)
# CHECK-INST: vlseg4e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 62 <unknown>

vlseg4e32.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 60 <unknown>

vlseg4e32.v v8, (a0)
# CHECK-INST: vlseg4e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 62 <unknown>

vlseg4e64.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 60 <unknown>

vlseg4e64.v v8, (a0)
# CHECK-INST: vlseg4e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 62 <unknown>

vlseg4e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 61 <unknown>

vlseg4e8ff.v v8, (a0)
# CHECK-INST: vlseg4e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 63 <unknown>

vlseg4e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 61 <unknown>

vlseg4e16ff.v v8, (a0)
# CHECK-INST: vlseg4e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 63 <unknown>

vlseg4e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 61 <unknown>

vlseg4e32ff.v v8, (a0)
# CHECK-INST: vlseg4e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 63 <unknown>

vlseg4e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 61 <unknown>

vlseg4e64ff.v v8, (a0)
# CHECK-INST: vlseg4e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 63 <unknown>

vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 68 <unknown>

vlsseg4e8.v v8, (a0), a1
# CHECK-INST: vlsseg4e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 6a <unknown>

vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 68 <unknown>

vlsseg4e16.v v8, (a0), a1
# CHECK-INST: vlsseg4e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 6a <unknown>

vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 68 <unknown>

vlsseg4e32.v v8, (a0), a1
# CHECK-INST: vlsseg4e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 6a <unknown>

vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 68 <unknown>

vlsseg4e64.v v8, (a0), a1
# CHECK-INST: vlsseg4e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 6a <unknown>

vluxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 64 <unknown>

vluxseg4ei8.v v8, (a0), v4
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 66 <unknown>

vluxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 64 <unknown>

vluxseg4ei16.v v8, (a0), v4
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 66 <unknown>

vluxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 64 <unknown>

vluxseg4ei32.v v8, (a0), v4
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 66 <unknown>

vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 64 <unknown>

vluxseg4ei64.v v8, (a0), v4
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 66 <unknown>

vloxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 6c <unknown>

vloxseg4ei8.v v8, (a0), v4
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 6e <unknown>

vloxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 6c <unknown>

vloxseg4ei16.v v8, (a0), v4
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 6e <unknown>

vloxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 6c <unknown>

vloxseg4ei32.v v8, (a0), v4
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 6e <unknown>

vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 6c <unknown>

vloxseg4ei64.v v8, (a0), v4
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 6e <unknown>

vlseg5e8.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 80 <unknown>

vlseg5e8.v v8, (a0)
# CHECK-INST: vlseg5e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 82 <unknown>

vlseg5e16.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 80 <unknown>

vlseg5e16.v v8, (a0)
# CHECK-INST: vlseg5e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 82 <unknown>

vlseg5e32.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 80 <unknown>

vlseg5e32.v v8, (a0)
# CHECK-INST: vlseg5e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 82 <unknown>

vlseg5e64.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 80 <unknown>

vlseg5e64.v v8, (a0)
# CHECK-INST: vlseg5e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 82 <unknown>

vlseg5e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 81 <unknown>

vlseg5e8ff.v v8, (a0)
# CHECK-INST: vlseg5e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 83 <unknown>

vlseg5e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 81 <unknown>

vlseg5e16ff.v v8, (a0)
# CHECK-INST: vlseg5e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 83 <unknown>

vlseg5e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 81 <unknown>

vlseg5e32ff.v v8, (a0)
# CHECK-INST: vlseg5e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 83 <unknown>

vlseg5e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 81 <unknown>

vlseg5e64ff.v v8, (a0)
# CHECK-INST: vlseg5e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 83 <unknown>

vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 88 <unknown>

vlsseg5e8.v v8, (a0), a1
# CHECK-INST: vlsseg5e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 8a <unknown>

vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 88 <unknown>

vlsseg5e16.v v8, (a0), a1
# CHECK-INST: vlsseg5e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 8a <unknown>

vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 88 <unknown>

vlsseg5e32.v v8, (a0), a1
# CHECK-INST: vlsseg5e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 8a <unknown>

vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 88 <unknown>

vlsseg5e64.v v8, (a0), a1
# CHECK-INST: vlsseg5e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 8a <unknown>

vluxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 84 <unknown>

vluxseg5ei8.v v8, (a0), v4
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 86 <unknown>

vluxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 84 <unknown>

vluxseg5ei16.v v8, (a0), v4
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 86 <unknown>

vluxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 84 <unknown>

vluxseg5ei32.v v8, (a0), v4
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 86 <unknown>

vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 84 <unknown>

vluxseg5ei64.v v8, (a0), v4
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 86 <unknown>

vloxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 8c <unknown>

vloxseg5ei8.v v8, (a0), v4
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 8e <unknown>

vloxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 8c <unknown>

vloxseg5ei16.v v8, (a0), v4
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 8e <unknown>

vloxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 8c <unknown>

vloxseg5ei32.v v8, (a0), v4
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 8e <unknown>

vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 8c <unknown>

vloxseg5ei64.v v8, (a0), v4
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 8e <unknown>

vlseg6e8.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 a0 <unknown>

vlseg6e8.v v8, (a0)
# CHECK-INST: vlseg6e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 a2 <unknown>

vlseg6e16.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 a0 <unknown>

vlseg6e16.v v8, (a0)
# CHECK-INST: vlseg6e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 a2 <unknown>

vlseg6e32.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 a0 <unknown>

vlseg6e32.v v8, (a0)
# CHECK-INST: vlseg6e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 a2 <unknown>

vlseg6e64.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 a0 <unknown>

vlseg6e64.v v8, (a0)
# CHECK-INST: vlseg6e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 a2 <unknown>

vlseg6e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 a1 <unknown>

vlseg6e8ff.v v8, (a0)
# CHECK-INST: vlseg6e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 a3 <unknown>

vlseg6e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 a1 <unknown>

vlseg6e16ff.v v8, (a0)
# CHECK-INST: vlseg6e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 a3 <unknown>

vlseg6e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 a1 <unknown>

vlseg6e32ff.v v8, (a0)
# CHECK-INST: vlseg6e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 a3 <unknown>

vlseg6e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 a1 <unknown>

vlseg6e64ff.v v8, (a0)
# CHECK-INST: vlseg6e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 a3 <unknown>

vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 a8 <unknown>

vlsseg6e8.v v8, (a0), a1
# CHECK-INST: vlsseg6e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 aa <unknown>

vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 a8 <unknown>

vlsseg6e16.v v8, (a0), a1
# CHECK-INST: vlsseg6e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 aa <unknown>

vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 a8 <unknown>

vlsseg6e32.v v8, (a0), a1
# CHECK-INST: vlsseg6e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 aa <unknown>

vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 a8 <unknown>

vlsseg6e64.v v8, (a0), a1
# CHECK-INST: vlsseg6e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 aa <unknown>

vluxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 a4 <unknown>

vluxseg6ei8.v v8, (a0), v4
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 a6 <unknown>

vluxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 a4 <unknown>

vluxseg6ei16.v v8, (a0), v4
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 a6 <unknown>

vluxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 a4 <unknown>

vluxseg6ei32.v v8, (a0), v4
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 a6 <unknown>

vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 a4 <unknown>

vluxseg6ei64.v v8, (a0), v4
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 a6 <unknown>

vloxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 ac <unknown>

vloxseg6ei8.v v8, (a0), v4
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 ae <unknown>

vloxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 ac <unknown>

vloxseg6ei16.v v8, (a0), v4
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 ae <unknown>

vloxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 ac <unknown>

vloxseg6ei32.v v8, (a0), v4
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 ae <unknown>

vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 ac <unknown>

vloxseg6ei64.v v8, (a0), v4
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 ae <unknown>

vlseg7e8.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 c0 <unknown>

vlseg7e8.v v8, (a0)
# CHECK-INST: vlseg7e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 c2 <unknown>

vlseg7e16.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 c0 <unknown>

vlseg7e16.v v8, (a0)
# CHECK-INST: vlseg7e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 c2 <unknown>

vlseg7e32.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 c0 <unknown>

vlseg7e32.v v8, (a0)
# CHECK-INST: vlseg7e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 c2 <unknown>

vlseg7e64.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 c0 <unknown>

vlseg7e64.v v8, (a0)
# CHECK-INST: vlseg7e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 c2 <unknown>

vlseg7e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 c1 <unknown>

vlseg7e8ff.v v8, (a0)
# CHECK-INST: vlseg7e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 c3 <unknown>

vlseg7e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 c1 <unknown>

vlseg7e16ff.v v8, (a0)
# CHECK-INST: vlseg7e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 c3 <unknown>

vlseg7e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 c1 <unknown>

vlseg7e32ff.v v8, (a0)
# CHECK-INST: vlseg7e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 c3 <unknown>

vlseg7e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 c1 <unknown>

vlseg7e64ff.v v8, (a0)
# CHECK-INST: vlseg7e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 c3 <unknown>

vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 c8 <unknown>

vlsseg7e8.v v8, (a0), a1
# CHECK-INST: vlsseg7e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 ca <unknown>

vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 c8 <unknown>

vlsseg7e16.v v8, (a0), a1
# CHECK-INST: vlsseg7e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 ca <unknown>

vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 c8 <unknown>

vlsseg7e32.v v8, (a0), a1
# CHECK-INST: vlsseg7e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 ca <unknown>

vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 c8 <unknown>

vlsseg7e64.v v8, (a0), a1
# CHECK-INST: vlsseg7e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 ca <unknown>

vluxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 c4 <unknown>

vluxseg7ei8.v v8, (a0), v4
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 c6 <unknown>

vluxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 c4 <unknown>

vluxseg7ei16.v v8, (a0), v4
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 c6 <unknown>

vluxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 c4 <unknown>

vluxseg7ei32.v v8, (a0), v4
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 c6 <unknown>

vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 c4 <unknown>

vluxseg7ei64.v v8, (a0), v4
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 c6 <unknown>

vloxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 cc <unknown>

vloxseg7ei8.v v8, (a0), v4
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 ce <unknown>

vloxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 cc <unknown>

vloxseg7ei16.v v8, (a0), v4
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 ce <unknown>

vloxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 cc <unknown>

vloxseg7ei32.v v8, (a0), v4
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 ce <unknown>

vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 cc <unknown>

vloxseg7ei64.v v8, (a0), v4
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 ce <unknown>

vlseg8e8.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 e0 <unknown>

vlseg8e8.v v8, (a0)
# CHECK-INST: vlseg8e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 e2 <unknown>

vlseg8e16.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 e0 <unknown>

vlseg8e16.v v8, (a0)
# CHECK-INST: vlseg8e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 e2 <unknown>

vlseg8e32.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 e0 <unknown>

vlseg8e32.v v8, (a0)
# CHECK-INST: vlseg8e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 e2 <unknown>

vlseg8e64.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 e0 <unknown>

vlseg8e64.v v8, (a0)
# CHECK-INST: vlseg8e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 e2 <unknown>

vlseg8e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 e1 <unknown>

vlseg8e8ff.v v8, (a0)
# CHECK-INST: vlseg8e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 05 e3 <unknown>

vlseg8e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 e1 <unknown>

vlseg8e16ff.v v8, (a0)
# CHECK-INST: vlseg8e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 05 e3 <unknown>

vlseg8e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 e1 <unknown>

vlseg8e32ff.v v8, (a0)
# CHECK-INST: vlseg8e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 05 e3 <unknown>

vlseg8e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 e1 <unknown>

vlseg8e64ff.v v8, (a0)
# CHECK-INST: vlseg8e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 05 e3 <unknown>

vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 e8 <unknown>

vlsseg8e8.v v8, (a0), a1
# CHECK-INST: vlsseg8e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 b5 ea <unknown>

vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 e8 <unknown>

vlsseg8e16.v v8, (a0), a1
# CHECK-INST: vlsseg8e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 b5 ea <unknown>

vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 e8 <unknown>

vlsseg8e32.v v8, (a0), a1
# CHECK-INST: vlsseg8e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 b5 ea <unknown>

vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 e8 <unknown>

vlsseg8e64.v v8, (a0), a1
# CHECK-INST: vlsseg8e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 b5 ea <unknown>

vluxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 e4 <unknown>

vluxseg8ei8.v v8, (a0), v4
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 e6 <unknown>

vluxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 e4 <unknown>

vluxseg8ei16.v v8, (a0), v4
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 e6 <unknown>

vluxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 e4 <unknown>

vluxseg8ei32.v v8, (a0), v4
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 e6 <unknown>

vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 e4 <unknown>

vluxseg8ei64.v v8, (a0), v4
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 e6 <unknown>

vloxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 ec <unknown>

vloxseg8ei8.v v8, (a0), v4
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 04 45 ee <unknown>

vloxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 ec <unknown>

vloxseg8ei16.v v8, (a0), v4
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 54 45 ee <unknown>

vloxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 ec <unknown>

vloxseg8ei32.v v8, (a0), v4
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 64 45 ee <unknown>

vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 ec <unknown>

vloxseg8ei64.v v8, (a0), v4
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 07 74 45 ee <unknown>

vsseg2e8.v v24, (a0), v0.t
# CHECK-INST: vsseg2e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 20 <unknown>

vsseg2e8.v v24, (a0)
# CHECK-INST: vsseg2e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 22 <unknown>

vsseg2e16.v v24, (a0), v0.t
# CHECK-INST: vsseg2e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 20 <unknown>

vsseg2e16.v v24, (a0)
# CHECK-INST: vsseg2e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 22 <unknown>

vsseg2e32.v v24, (a0), v0.t
# CHECK-INST: vsseg2e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 20 <unknown>

vsseg2e32.v v24, (a0)
# CHECK-INST: vsseg2e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 22 <unknown>

vsseg2e64.v v24, (a0), v0.t
# CHECK-INST: vsseg2e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 20 <unknown>

vsseg2e64.v v24, (a0)
# CHECK-INST: vsseg2e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 22 <unknown>

vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 28 <unknown>

vssseg2e8.v v24, (a0), a1
# CHECK-INST: vssseg2e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 2a <unknown>

vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 28 <unknown>

vssseg2e16.v v24, (a0), a1
# CHECK-INST: vssseg2e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 2a <unknown>

vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 28 <unknown>

vssseg2e32.v v24, (a0), a1
# CHECK-INST: vssseg2e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 2a <unknown>

vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 28 <unknown>

vssseg2e64.v v24, (a0), a1
# CHECK-INST: vssseg2e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 2a <unknown>

vsuxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 24 <unknown>

vsuxseg2ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 26 <unknown>

vsuxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 24 <unknown>

vsuxseg2ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 26 <unknown>

vsuxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 24 <unknown>

vsuxseg2ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 26 <unknown>

vsuxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 24 <unknown>

vsuxseg2ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 26 <unknown>

vsoxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 2c <unknown>

vsoxseg2ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 2e <unknown>

vsoxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 2c <unknown>

vsoxseg2ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 2e <unknown>

vsoxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 2c <unknown>

vsoxseg2ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 2e <unknown>

vsoxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 2c <unknown>

vsoxseg2ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 2e <unknown>

vsseg3e8.v v24, (a0), v0.t
# CHECK-INST: vsseg3e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 40 <unknown>

vsseg3e8.v v24, (a0)
# CHECK-INST: vsseg3e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 42 <unknown>

vsseg3e16.v v24, (a0), v0.t
# CHECK-INST: vsseg3e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 40 <unknown>

vsseg3e16.v v24, (a0)
# CHECK-INST: vsseg3e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 42 <unknown>

vsseg3e32.v v24, (a0), v0.t
# CHECK-INST: vsseg3e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 40 <unknown>

vsseg3e32.v v24, (a0)
# CHECK-INST: vsseg3e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 42 <unknown>

vsseg3e64.v v24, (a0), v0.t
# CHECK-INST: vsseg3e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 40 <unknown>

vsseg3e64.v v24, (a0)
# CHECK-INST: vsseg3e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 42 <unknown>

vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 48 <unknown>

vssseg3e8.v v24, (a0), a1
# CHECK-INST: vssseg3e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 4a <unknown>

vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 48 <unknown>

vssseg3e16.v v24, (a0), a1
# CHECK-INST: vssseg3e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 4a <unknown>

vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 48 <unknown>

vssseg3e32.v v24, (a0), a1
# CHECK-INST: vssseg3e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 4a <unknown>

vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 48 <unknown>

vssseg3e64.v v24, (a0), a1
# CHECK-INST: vssseg3e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 4a <unknown>

vsuxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 44 <unknown>

vsuxseg3ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 46 <unknown>

vsuxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 44 <unknown>

vsuxseg3ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 46 <unknown>

vsuxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 44 <unknown>

vsuxseg3ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 46 <unknown>

vsuxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 44 <unknown>

vsuxseg3ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 46 <unknown>

vsoxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 4c <unknown>

vsoxseg3ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 4e <unknown>

vsoxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 4c <unknown>

vsoxseg3ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 4e <unknown>

vsoxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 4c <unknown>

vsoxseg3ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 4e <unknown>

vsoxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 4c <unknown>

vsoxseg3ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 4e <unknown>

vsseg4e8.v v24, (a0), v0.t
# CHECK-INST: vsseg4e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 60 <unknown>

vsseg4e8.v v24, (a0)
# CHECK-INST: vsseg4e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 62 <unknown>

vsseg4e16.v v24, (a0), v0.t
# CHECK-INST: vsseg4e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 60 <unknown>

vsseg4e16.v v24, (a0)
# CHECK-INST: vsseg4e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 62 <unknown>

vsseg4e32.v v24, (a0), v0.t
# CHECK-INST: vsseg4e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 60 <unknown>

vsseg4e32.v v24, (a0)
# CHECK-INST: vsseg4e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 62 <unknown>

vsseg4e64.v v24, (a0), v0.t
# CHECK-INST: vsseg4e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 60 <unknown>

vsseg4e64.v v24, (a0)
# CHECK-INST: vsseg4e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 62 <unknown>

vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 68 <unknown>

vssseg4e8.v v24, (a0), a1
# CHECK-INST: vssseg4e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 6a <unknown>

vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 68 <unknown>

vssseg4e16.v v24, (a0), a1
# CHECK-INST: vssseg4e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 6a <unknown>

vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 68 <unknown>

vssseg4e32.v v24, (a0), a1
# CHECK-INST: vssseg4e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 6a <unknown>

vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 68 <unknown>

vssseg4e64.v v24, (a0), a1
# CHECK-INST: vssseg4e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 6a <unknown>

vsuxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 64 <unknown>

vsuxseg4ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 66 <unknown>

vsuxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 64 <unknown>

vsuxseg4ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 66 <unknown>

vsuxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 64 <unknown>

vsuxseg4ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 66 <unknown>

vsuxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 64 <unknown>

vsuxseg4ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 66 <unknown>

vsoxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 6c <unknown>

vsoxseg4ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 6e <unknown>

vsoxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 6c <unknown>

vsoxseg4ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 6e <unknown>

vsoxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 6c <unknown>

vsoxseg4ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 6e <unknown>

vsoxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 6c <unknown>

vsoxseg4ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 6e <unknown>

vsseg5e8.v v24, (a0), v0.t
# CHECK-INST: vsseg5e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 80 <unknown>

vsseg5e8.v v24, (a0)
# CHECK-INST: vsseg5e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 82 <unknown>

vsseg5e16.v v24, (a0), v0.t
# CHECK-INST: vsseg5e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 80 <unknown>

vsseg5e16.v v24, (a0)
# CHECK-INST: vsseg5e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 82 <unknown>

vsseg5e32.v v24, (a0), v0.t
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 80 <unknown>

vsseg5e32.v v24, (a0)
# CHECK-INST: vsseg5e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 82 <unknown>

vsseg5e64.v v24, (a0), v0.t
# CHECK-INST: vsseg5e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 80 <unknown>

vsseg5e64.v v24, (a0)
# CHECK-INST: vsseg5e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 82 <unknown>

vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 88 <unknown>

vssseg5e8.v v24, (a0), a1
# CHECK-INST: vssseg5e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 8a <unknown>

vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 88 <unknown>

vssseg5e16.v v24, (a0), a1
# CHECK-INST: vssseg5e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 8a <unknown>

vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 88 <unknown>

vssseg5e32.v v24, (a0), a1
# CHECK-INST: vssseg5e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 8a <unknown>

vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 88 <unknown>

vssseg5e64.v v24, (a0), a1
# CHECK-INST: vssseg5e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 8a <unknown>

vsuxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 84 <unknown>

vsuxseg5ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 86 <unknown>

vsuxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 84 <unknown>

vsuxseg5ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 86 <unknown>

vsuxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 84 <unknown>

vsuxseg5ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 86 <unknown>

vsuxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 84 <unknown>

vsuxseg5ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 86 <unknown>

vsoxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 8c <unknown>

vsoxseg5ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 8e <unknown>

vsoxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 8c <unknown>

vsoxseg5ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 8e <unknown>

vsoxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 8c <unknown>

vsoxseg5ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 8e <unknown>

vsoxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 8c <unknown>

vsoxseg5ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 8e <unknown>

vsseg6e8.v v24, (a0), v0.t
# CHECK-INST: vsseg6e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 a0 <unknown>

vsseg6e8.v v24, (a0)
# CHECK-INST: vsseg6e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 a2 <unknown>

vsseg6e16.v v24, (a0), v0.t
# CHECK-INST: vsseg6e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 a0 <unknown>

vsseg6e16.v v24, (a0)
# CHECK-INST: vsseg6e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 a2 <unknown>

vsseg6e32.v v24, (a0), v0.t
# CHECK-INST: vsseg6e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 a0 <unknown>

vsseg6e32.v v24, (a0)
# CHECK-INST: vsseg6e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 a2 <unknown>

vsseg6e64.v v24, (a0), v0.t
# CHECK-INST: vsseg6e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 a0 <unknown>

vsseg6e64.v v24, (a0)
# CHECK-INST: vsseg6e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 a2 <unknown>

vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 a8 <unknown>

vssseg6e8.v v24, (a0), a1
# CHECK-INST: vssseg6e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 aa <unknown>

vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 a8 <unknown>

vssseg6e16.v v24, (a0), a1
# CHECK-INST: vssseg6e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 aa <unknown>

vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 a8 <unknown>

vssseg6e32.v v24, (a0), a1
# CHECK-INST: vssseg6e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 aa <unknown>

vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 a8 <unknown>

vssseg6e64.v v24, (a0), a1
# CHECK-INST: vssseg6e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 aa <unknown>

vsuxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 a4 <unknown>

vsuxseg6ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 a6 <unknown>

vsuxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 a4 <unknown>

vsuxseg6ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 a6 <unknown>

vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 a4 <unknown>

vsuxseg6ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 a6 <unknown>

vsuxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 a4 <unknown>

vsuxseg6ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 a6 <unknown>

vsoxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 ac <unknown>

vsoxseg6ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 ae <unknown>

vsoxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 ac <unknown>

vsoxseg6ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 ae <unknown>

vsoxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 ac <unknown>

vsoxseg6ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 ae <unknown>

vsoxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 ac <unknown>

vsoxseg6ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 ae <unknown>

vsseg7e8.v v24, (a0), v0.t
# CHECK-INST: vsseg7e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 c0 <unknown>

vsseg7e8.v v24, (a0)
# CHECK-INST: vsseg7e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 c2 <unknown>

vsseg7e16.v v24, (a0), v0.t
# CHECK-INST: vsseg7e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 c0 <unknown>

vsseg7e16.v v24, (a0)
# CHECK-INST: vsseg7e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 c2 <unknown>

vsseg7e32.v v24, (a0), v0.t
# CHECK-INST: vsseg7e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 c0 <unknown>

vsseg7e32.v v24, (a0)
# CHECK-INST: vsseg7e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 c2 <unknown>

vsseg7e64.v v24, (a0), v0.t
# CHECK-INST: vsseg7e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 c0 <unknown>

vsseg7e64.v v24, (a0)
# CHECK-INST: vsseg7e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 c2 <unknown>

vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 c8 <unknown>

vssseg7e8.v v24, (a0), a1
# CHECK-INST: vssseg7e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 ca <unknown>

vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 c8 <unknown>

vssseg7e16.v v24, (a0), a1
# CHECK-INST: vssseg7e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 ca <unknown>

vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 c8 <unknown>

vssseg7e32.v v24, (a0), a1
# CHECK-INST: vssseg7e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 ca <unknown>

vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 c8 <unknown>

vssseg7e64.v v24, (a0), a1
# CHECK-INST: vssseg7e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 ca <unknown>

vsuxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 c4 <unknown>

vsuxseg7ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 c6 <unknown>

vsuxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 c4 <unknown>

vsuxseg7ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 c6 <unknown>

vsuxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 c4 <unknown>

vsuxseg7ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 c6 <unknown>

vsuxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 c4 <unknown>

vsuxseg7ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 c6 <unknown>

vsoxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 cc <unknown>

vsoxseg7ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 ce <unknown>

vsoxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 cc <unknown>

vsoxseg7ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 ce <unknown>

vsoxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 cc <unknown>

vsoxseg7ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 ce <unknown>

vsoxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 cc <unknown>

vsoxseg7ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 ce <unknown>

vsseg8e8.v v24, (a0), v0.t
# CHECK-INST: vsseg8e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 e0 <unknown>

vsseg8e8.v v24, (a0)
# CHECK-INST: vsseg8e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 05 e2 <unknown>

vsseg8e16.v v24, (a0), v0.t
# CHECK-INST: vsseg8e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 e0 <unknown>

vsseg8e16.v v24, (a0)
# CHECK-INST: vsseg8e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 05 e2 <unknown>

vsseg8e32.v v24, (a0), v0.t
# CHECK-INST: vsseg8e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 e0 <unknown>

vsseg8e32.v v24, (a0)
# CHECK-INST: vsseg8e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 05 e2 <unknown>

vsseg8e64.v v24, (a0), v0.t
# CHECK-INST: vsseg8e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 e0 <unknown>

vsseg8e64.v v24, (a0)
# CHECK-INST: vsseg8e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 05 e2 <unknown>

vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 e8 <unknown>

vssseg8e8.v v24, (a0), a1
# CHECK-INST: vssseg8e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c b5 ea <unknown>

vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 e8 <unknown>

vssseg8e16.v v24, (a0), a1
# CHECK-INST: vssseg8e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c b5 ea <unknown>

vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 e8 <unknown>

vssseg8e32.v v24, (a0), a1
# CHECK-INST: vssseg8e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c b5 ea <unknown>

vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 e8 <unknown>

vssseg8e64.v v24, (a0), a1
# CHECK-INST: vssseg8e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c b5 ea <unknown>

vsuxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 e4 <unknown>

vsuxseg8ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 e6 <unknown>

vsuxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 e4 <unknown>

vsuxseg8ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 e6 <unknown>

vsuxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 e4 <unknown>

vsuxseg8ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 e6 <unknown>

vsuxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 e4 <unknown>

vsuxseg8ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 e6 <unknown>

vsoxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 ec <unknown>

vsoxseg8ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 0c 45 ee <unknown>

vsoxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 ec <unknown>

vsoxseg8ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 5c 45 ee <unknown>

vsoxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 ec <unknown>

vsoxseg8ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 6c 45 ee <unknown>

vsoxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 ec <unknown>

vsoxseg8ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvlsseg'
# CHECK-UNKNOWN: 27 7c 45 ee <unknown>
