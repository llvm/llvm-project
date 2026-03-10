# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | llvm-objdump -d  --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | llvm-objdump -d  --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a32f,+xsfmm32a8i,+xsfmm32a8f,+xsfmm64a64f %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: sf.vsettnt a0, a1, e8, w1
# CHECK-ENCODING: [0x57,0xf5,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 2005f557 <unknown>
sf.vsettnt a0, a1, e8, w1

# CHECK-INST: sf.vsettnt a0, a1, e16alt, w1
# CHECK-ENCODING: [0x57,0xf5,0x85,0x30]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 3085f557 <unknown>
sf.vsettnt a0, a1, e16alt, w1

# CHECK-INST: sf.vsettnt a0, a1, e8, w1
# CHECK-ENCODING: [0x57,0xf5,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 2005f557 <unknown>
vsetvli a0, a1, 0x200

# CHECK-INST: sf.vsettnt a0, a1, e16alt, w1
# CHECK-ENCODING: [0x57,0xf5,0x85,0x30]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 3085f557 <unknown>
vsetvli a0, a1, 0x308

# CHECK-INST: sf.vsettn a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x05,0x84]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 8405f557 <unknown>
sf.vsettn a0, a1

# CHECK-INST: sf.vsettm a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x15,0x84]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 8415f557 <unknown>
sf.vsettm a0, a1

# CHECK-INST: sf.vsettk a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x25,0x84]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 8425f557 <unknown>
sf.vsettk a0, a1

# CHECK-INST: sf.vlte8  a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x12]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 12a5f007 <unknown>
sf.vlte8  a0, (a1)

# CHECK-INST: sf.vlte16 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x32]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 32a5f007 <unknown>
sf.vlte16 a0, (a1)

# CHECK-INST: sf.vlte32 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x52]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 52a5f007 <unknown>
sf.vlte32 a0, (a1)

# CHECK-INST: sf.vlte64 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x72]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 72a5f007 <unknown>
sf.vlte64 a0, (a1)

# CHECK-INST: sf.vste8  a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x12]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 12a5f027 <unknown>
sf.vste8  a0, (a1)

# CHECK-INST: sf.vste16 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x32]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 32a5f027 <unknown>
sf.vste16 a0, (a1)

# CHECK-INST: sf.vste32 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x52]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 52a5f027 <unknown>
sf.vste32 a0, (a1)

# CHECK-INST: sf.vste64 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x72]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 72a5f027 <unknown>
sf.vste64 a0, (a1)

# CHECK-INST: sf.vtmv.v.t v8, a0
# CHECK-ENCODING: [0x57,0x64,0xf5,0x43]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 43f56457 <unknown>
sf.vtmv.v.t v8, a0

# CHECK-INST: sf.vtmv.t.v a0, v8
# CHECK-ENCODING: [0x57,0x60,0x85,0x5e]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 5e856057 <unknown>
sf.vtmv.t.v a0, v8

# CHECK-INST: sf.mm.f.f mt2, v8, v9
# CHECK-ENCODING: [0x77,0x92,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a16f' (TEW=32-bit accumulation, operands - float: 16b, widen=2 (IEEE, BF)), or 'XSfmm32a32f' (TEW=32-bit accumulation, operands - float: 32b), or 'XSfmm64a64f' (TEW=64-bit accumulation, operands - float: fp64){{$}}
# CHECK-UNKNOWN: f2849277 <unknown>
sf.mm.f.f mt2, v8, v9

# CHECK-INST: sf.mm.e5m2.e5m2 mt0, v8, v9
# CHECK-ENCODING: [0x77,0x90,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation, operands - float: fp8){{$}}
# CHECK-UNKNOWN: fa849077 <unknown>
sf.mm.e5m2.e5m2 mt0, v8, v9

# CHECK-INST: sf.mm.e5m2.e4m3 mt4, v8, v9
# CHECK-ENCODING: [0xf7,0x94,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation, operands - float: fp8){{$}}
# CHECK-UNKNOWN: fa8494f7 <unknown>
sf.mm.e5m2.e4m3 mt4, v8, v9

# CHECK-INST: sf.mm.e4m3.e5m2 mt8, v8, v9
# CHECK-ENCODING: [0x77,0x98,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation, operands - float: fp8){{$}}
# CHECK-UNKNOWN: fe849877 <unknown>
sf.mm.e4m3.e5m2 mt8, v8, v9

# CHECK-INST: sf.mm.e4m3.e4m3 mt12, v8, v9
# CHECK-ENCODING: [0xf7,0x9c,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation, operands - float: fp8){{$}}
# CHECK-UNKNOWN: fe849cf7 <unknown>
sf.mm.e4m3.e4m3 mt12, v8, v9

# CHECK-INST: sf.mm.u.u mt0, v8, v9
# CHECK-ENCODING: [0x77,0x80,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8i' (TEW=32-bit accumulation, operands - int: 8b){{$}}
# CHECK-UNKNOWN: f2848077 <unknown>
sf.mm.u.u mt0, v8, v9

# CHECK-INST: sf.mm.s.u mt4, v8, v9
# CHECK-ENCODING: [0x77,0x84,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8i' (TEW=32-bit accumulation, operands - int: 8b){{$}}
# CHECK-UNKNOWN: f6848477 <unknown>
sf.mm.s.u mt4, v8, v9

# CHECK-INST: sf.mm.u.s mt8, v8, v9
# CHECK-ENCODING: [0xf7,0x88,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8i' (TEW=32-bit accumulation, operands - int: 8b){{$}}
# CHECK-UNKNOWN: f28488f7 <unknown>
sf.mm.u.s mt8, v8, v9

# CHECK-INST: sf.mm.s.s mt12, v8, v9
# CHECK-ENCODING: [0xf7,0x8c,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8i' (TEW=32-bit accumulation, operands - int: 8b){{$}}
# CHECK-UNKNOWN: f6848cf7 <unknown>
sf.mm.s.s mt12, v8, v9

# CHECK-INST: sf.vtzero.t mt15
# CHECK-ENCODING: [0x57,0x6f,0xe0,0x43]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero){{$}}
# CHECK-UNKNOWN: 43e06f57 <unknown>
sf.vtzero.t mt15

# CHECK-INST: vsetvl a2, a0, a1
# CHECK-ENCODING: [0x57,0x76,0xb5,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 80b57657 <unknown>
vsetvl a2, a0, a1

# CHECK-INST:  vsetvli a0, a1, e8, m1, tu, mu
# CHECK-ENCODING: [0x57,0xf5,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 0005f557 <unknown>
vsetvli a0, a1, e8, m1, tu, mu
