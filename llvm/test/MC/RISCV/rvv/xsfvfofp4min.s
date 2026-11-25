# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+xsfvfofp4min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+xsfvfofp4min %s \
# RUN:    | llvm-objdump -d --mattr=+xsfvfofp4min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+xsfvfofp4min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+xsfvfofp4min %s \
# RUN:    | llvm-objdump -d --mattr=+xsfvfofp4min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST


# CHECK-INST: sf.vfext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x4b,0x4a]
sf.vfext.vf2 v8, v4 # expected-error {{instruction requires the following: 'XSfvfofp4min' (SiFive Vector OFP4 Conversion Instruction)}}

# CHECK-INST: sf.vfext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4b,0x48]
sf.vfext.vf2 v8, v4, v0.t # expected-error {{instruction requires the following: 'XSfvfofp4min' (SiFive Vector OFP4 Conversion Instruction)}}
