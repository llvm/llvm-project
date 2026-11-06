# XAndesVPackFPH - Andes Vector Packed FP16 Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvpackfph -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvpackfph < %s \
# RUN:     | llvm-objdump --mattr=+xandesvpackfph -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvpackfph -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvpackfph < %s \
# RUN:     | llvm-objdump --mattr=+xandesvpackfph -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s

# CHECK-OBJ: nds.vfpmadt.vf v8, fa0, v10
# CHECK-ASM: nds.vfpmadt.vf v8, fa0, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa5,0x0a]
# expected-error@+1 {{instruction requires the following: 'XAndesVPackFPH' (Andes Vector Packed FP16 Extension)}}
nds.vfpmadt.vf v8, fa0, v10

# CHECK-OBJ: nds.vfpmadt.vf v8, fa0, v10, v0.t
# CHECK-ASM: nds.vfpmadt.vf v8, fa0, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa5,0x08]
# expected-error@+1 {{instruction requires the following: 'XAndesVPackFPH' (Andes Vector Packed FP16 Extension)}}
nds.vfpmadt.vf v8, fa0, v10, v0.t

# CHECK-OBJ: nds.vfpmadb.vf v8, fa0, v10
# CHECK-ASM: nds.vfpmadb.vf v8, fa0, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa5,0x0e]
# expected-error@+1 {{instruction requires the following: 'XAndesVPackFPH' (Andes Vector Packed FP16 Extension)}}
nds.vfpmadb.vf v8, fa0, v10

# CHECK-OBJ: nds.vfpmadb.vf v8, fa0, v10, v0.t
# CHECK-ASM: nds.vfpmadb.vf v8, fa0, v10, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xa5,0x0c]
# expected-error@+1 {{instruction requires the following: 'XAndesVPackFPH' (Andes Vector Packed FP16 Extension)}}
nds.vfpmadb.vf v8, fa0, v10, v0.t
