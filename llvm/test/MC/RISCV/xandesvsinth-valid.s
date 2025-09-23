# XAndesVSIntLoad - Andes Vector INT4 Load Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvsinth -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvsinth < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsinth -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvsinth -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvsinth < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsinth -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# CHECK-OBJ: nds.vfwcvt.f.n.v  v8, v10
# CHECK-ASM: nds.vfwcvt.f.n.v  v8, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa2,0x02]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.n.v v8, v10
# CHECK-OBJ: nds.vfwcvt.f.n.v  v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.n.v  v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa2,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.n.v v8, v10, v0.t
# CHECK-OBJ: nds.vfwcvt.f.nu.v v8, v10
# CHECK-ASM: nds.vfwcvt.f.nu.v v8, v10
# CHECK-ASM: encoding: [0x5b,0xc4,0xa2,0x02]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.nu.v v8, v10
# CHECK-OBJ: nds.vfwcvt.f.nu.v v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.nu.v v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0xc4,0xa2,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.nu.v v8, v10, v0.t
# CHECK-OBJ: nds.vfwcvt.f.b.v  v8, v10
# CHECK-ASM: nds.vfwcvt.f.b.v  v8, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa3,0x02]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.b.v v8, v10
# CHECK-OBJ: nds.vfwcvt.f.b.v  v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.b.v  v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa3,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.b.v v8, v10, v0.t
# CHECK-OBJ: nds.vfwcvt.f.bu.v v8, v10
# CHECK-ASM: nds.vfwcvt.f.bu.v v8, v10
# CHECK-ASM: encoding: [0x5b,0xc4,0xa3,0x02]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.bu.v v8, v10
# CHECK-OBJ: nds.vfwcvt.f.bu.v v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.bu.v v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0xc4,0xa3,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vfwcvt.f.bu.v v8, v10, v0.t
# CHECK-OBJ: nds.vle4.v      v8, (a0)
# CHECK-ASM: nds.vle4.v      v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x05,0x06]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension){{$}}
nds.vle4.v v8, (a0)
