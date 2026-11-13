# XAndesVSIntH - Andes Vector Small INT Handling Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvsinth,+zve32f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvsinth,+zve32f < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsinth,+zve32f -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zve32f -show-encoding %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvsinth,+zve32f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvsinth,+zve32f < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsinth,+zve32f -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zve32f -show-encoding %s
# CHECK-OBJ: nds.vfwcvt.f.n.v  v8, v10
# CHECK-ASM: nds.vfwcvt.f.n.v  v8, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa2,0x02]
nds.vfwcvt.f.n.v v8, v10 # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.n.v  v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.n.v  v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa2,0x00]
nds.vfwcvt.f.n.v v8, v10, v0.t # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.nu.v v8, v10
# CHECK-ASM: nds.vfwcvt.f.nu.v v8, v10
# CHECK-ASM: encoding: [0x5b,0xc4,0xa2,0x02]
nds.vfwcvt.f.nu.v v8, v10 # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.nu.v v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.nu.v v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0xc4,0xa2,0x00]
nds.vfwcvt.f.nu.v v8, v10, v0.t # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.b.v  v8, v10
# CHECK-ASM: nds.vfwcvt.f.b.v  v8, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa3,0x02]
nds.vfwcvt.f.b.v v8, v10 # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.b.v  v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.b.v  v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa3,0x00]
nds.vfwcvt.f.b.v v8, v10, v0.t # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.bu.v v8, v10
# CHECK-ASM: nds.vfwcvt.f.bu.v v8, v10
# CHECK-ASM: encoding: [0x5b,0xc4,0xa3,0x02]
nds.vfwcvt.f.bu.v v8, v10 # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vfwcvt.f.bu.v v8, v10, v0.t
# CHECK-ASM: nds.vfwcvt.f.bu.v v8, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0xc4,0xa3,0x00]
nds.vfwcvt.f.bu.v v8, v10, v0.t # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
# CHECK-OBJ: nds.vle4.v      v8, (a0)
# CHECK-ASM: nds.vle4.v      v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x05,0x06]
nds.vle4.v v8, (a0) # expected-error {{instruction requires the following: 'XAndesVSIntH' (Andes Vector Small INT Handling Extension)}}
