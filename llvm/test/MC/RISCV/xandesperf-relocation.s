# RUN: llvm-mc -triple riscv32 -mattr=+xandesperf -M no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump -dr --mattr=+xandesperf - \
# RUN:     | FileCheck -check-prefix=OBJ %s
# RUN: llvm-mc -triple riscv64 -mattr=+xandesperf -M no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump -dr --mattr=+xandesperf - \
# RUN:     | FileCheck -check-prefix=OBJ %s

.long foo

# ASM: nds.bbc t0, 7, foo
# OBJ: nds.bbc t0, 0x7, 0x4 <.text+0x4>
# OBJ-NEXT: R_RISCV_VENDOR ANDES{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM241 foo{{$}}
nds.bbc t0, 7, foo

# ASM: nds.bbs t0, 7, foo
# OBJ-NEXT: nds.bbs t0, 0x7, 0x8 <.text+0x8>
# OBJ-NEXT: R_RISCV_VENDOR ANDES{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM241 foo{{$}}
nds.bbs t0, 7, foo

# ASM: nds.beqc t0, 7, foo
# OBJ-NEXT: nds.beqc t0, 0x7, 0xc <.text+0xc>
# OBJ-NEXT: R_RISCV_VENDOR ANDES{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM241 foo{{$}}
nds.beqc t0, 7, foo

# ASM: nds.bnec t0, 7, foo
# OBJ-NEXT: nds.bnec t0, 0x7, 0x10 <.text+0x10>
# OBJ-NEXT: R_RISCV_VENDOR ANDES{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM241 foo{{$}}
nds.bnec t0, 7, foo
