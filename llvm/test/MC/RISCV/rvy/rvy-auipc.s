# RUN: llvm-mc --triple=riscv32 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --triple=riscv32 --mattr=+experimental-y,-cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT,CHECK-INT-32 %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,-cap-mode --riscv-no-aliases --show-encoding --show-inst < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT,CHECK-INT-64 %s
## Check that the capmode output can be dissassembled correctly (this previous triggered an assertion in RISCVMCInstrAnalysis)
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y,+cap-mode < %s -o - \
# RUN:   | llvm-objdump --mattr=+experimental-y,-cap-mode -M no-aliases -d -r --no-print-imm-hex - \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-OBJ %s

# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --show-inst --defsym=RV64=1 < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-CAP %s
# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y,-cap-mode --riscv-no-aliases --show-encoding --show-inst --defsym=RV64=1 < %s \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-ASM,CHECK-INT,CHECK-INT-64 %s
# DISABLED: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y,+cap-mode --defsym=RV64=1 < %s -o %t64.o
# DISABLED: llvm-objdump --mattr=+experimental-y,-cap-mode -M no-aliases -d -r --no-print-imm-hex %t64.o \
# DISABLED:   | FileCheck --check-prefixes=CHECK,CHECK-OBJ %s
# CHECK: auipc		a0, 0
# CHECK-ASM-SAME: # encoding: [0x17,0x05,0x00,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:0>>
auipc a0, 0
# CHECK-NEXT: auipc		a0, 1
# CHECK-ASM-SAME: # encoding: [0x17,0x15,0x00,0x00]
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Imm:1>>
auipc a0, 1
# CHECK-ASM-NEXT: auipc		a0, %pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0bAAAA0101,A,A]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %pcrel_hi(sym), kind: fixup_riscv_pcrel_hi20
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_hi(sym)>>
# CHECK-OBJ-NEXT: auipc		a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_HI20 sym
auipc a0, %pcrel_hi(sym)
#
## Test the pseudo expansions using AUIPC:
#
# CHECK-OBJ-EMPTY:
# CHECK-NEXT: .Lpcrel_hi0
# CHECK-ASM-NEXT: auipc		a0, %pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0bAAAA0101,A,A]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %pcrel_hi(sym), kind: fixup_riscv_pcrel_hi20
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_hi(sym)>>
# CHECK-INT-NEXT: addi		a0, a0, %pcrel_lo(.Lpcrel_hi0)
# CHECK-INT-SAME: # encoding: [0x13,0x05,0bAAAA0101,A]
# CHECK-INT-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi0), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-NEXT: # <MCInst #[[#]] ADDI{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: addiy		a0, a0, %pcrel_lo(.Lpcrel_hi0)
# CHECK-CAP-SAME: # encoding: [0x1b,0x25,0bAAAA0101,A]
# CHECK-CAP-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi0), kind: fixup_riscv_pcrel_lo12_i
# CHECK-CAP-NEXT: # <MCInst #[[#]] ADDIY{{$}}
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_lo(.Lpcrel_hi0)>>
# CHECK-OBJ-NEXT: auipc		a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_HI20 sym
# CHECK-OBJ-NEXT: addiy a0, a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi0
la a0, sym
# CHECK-OBJ-EMPTY:
# CHECK-NEXT: .Lpcrel_hi1
# CHECK-ASM-NEXT: auipc		a0, %pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0bAAAA0101,A,A]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %pcrel_hi(sym), kind: fixup_riscv_pcrel_hi20
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_hi(sym)>>
# CHECK-INT-NEXT: addi		a0, a0, %pcrel_lo(.Lpcrel_hi1)
# CHECK-INT-SAME: # encoding: [0x13,0x05,0bAAAA0101,A]
# CHECK-INT-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi1), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-NEXT: # <MCInst #[[#]] ADDI{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: addiy		a0, a0, %pcrel_lo(.Lpcrel_hi1)
# CHECK-CAP-SAME: # encoding: [0x1b,0x25,0bAAAA0101,A]
# CHECK-CAP-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi1), kind: fixup_riscv_pcrel_lo12_i
# CHECK-CAP-NEXT: # <MCInst #[[#]] ADDIY{{$}}
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_lo(.Lpcrel_hi1)>>
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_HI20 sym
# CHECK-OBJ-NEXT: addiy a0, a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi1
lla a0, sym
# CHECK-OBJ-EMPTY:
# CHECK-NEXT: .Lpcrel_hi2
# CHECK-ASM-NEXT: auipc		a0, %got_pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0x05,0x00,0x00]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %got_pcrel_hi(sym), relocation type: 20
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%got_pcrel_hi(sym)>>
# CHECK-INT-32-NEXT: lw		a0, %pcrel_lo(.Lpcrel_hi2)(a0)
# CHECK-INT-32-SAME: # encoding: [0x03,0x25,0bAAAA0101,A]
# CHECK-INT-32-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi2), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-32-NEXT: # <MCInst #[[#]] LW{{$}}
# CHECK-INT-64-NEXT: ld		a0, %pcrel_lo(.Lpcrel_hi2)(a0)
# CHECK-INT-64-SAME: # encoding: [0x03,0x35,0bAAAA0101,A]
# CHECK-INT-64-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi2), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-64-NEXT: # <MCInst #[[#]] LD{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: ly		a0, %pcrel_lo(.Lpcrel_hi2)(a0)
# CHECK-CAP-SAME: # encoding: [0x0f,0x45,0bAAAA0101,A]
# CHECK-CAP-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi2), kind: fixup_riscv_pcrel_lo12_i
# CHECK-CAP-NEXT: # <MCInst #[[#]] LY{{$}}
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_lo(.Lpcrel_hi2)>>
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_GOT_HI20 sym
# CHECK-OBJ-NEXT: ly a0, 0(a0)
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi2
lga a0, sym
# CHECK-OBJ-EMPTY:
# CHECK-NEXT: .Lpcrel_hi3
# CHECK-ASM-NEXT: auipc		a0, %tls_ie_pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0x05,0x00,0x00]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %tls_ie_pcrel_hi(sym), relocation type: 21
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%tls_ie_pcrel_hi(sym)>>
# CHECK-INT-32-NEXT: lw		a0, %pcrel_lo(.Lpcrel_hi3)(a0)
# CHECK-INT-32-SAME: # encoding: [0x03,0x25,0bAAAA0101,A]
# CHECK-INT-32-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi3), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-32-NEXT: # <MCInst #[[#]] LW{{$}}
# CHECK-INT-64-NEXT: ld		a0, %pcrel_lo(.Lpcrel_hi3)(a0)
# CHECK-INT-64-SAME: # encoding: [0x03,0x35,0bAAAA0101,A]
# CHECK-INT-64-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi3), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-64-NEXT: # <MCInst #[[#]] LD{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: ly		a0, %pcrel_lo(.Lpcrel_hi3)(a0)
# CHECK-CAP-SAME: # encoding: [0x0f,0x45,0bAAAA0101,A]
# CHECK-CAP-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi3), kind: fixup_riscv_pcrel_lo12_i
# CHECK-CAP-NEXT: # <MCInst #[[#]] LY{{$}}
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_lo(.Lpcrel_hi3)>>
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_TLS_GOT_HI20 sym
# CHECK-OBJ-NEXT: ly a0, 0(a0)
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi3
la.tls.ie a0, sym
# CHECK-OBJ-EMPTY:
# CHECK-NEXT: .Lpcrel_hi4
# CHECK-ASM-NEXT: auipc		a0, %tls_gd_pcrel_hi(sym)
# CHECK-ASM-SAME: # encoding: [0x17,0x05,0x00,0x00]
# CHECK-ASM-NEXT: #   fixup A - offset: 0, value: %tls_gd_pcrel_hi(sym), relocation type: 22
# CHECK-ASM-NEXT: # <MCInst #[[#]] AUIPC{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%tls_gd_pcrel_hi(sym)>>
# CHECK-INT-NEXT: addi		a0, a0, %pcrel_lo(.Lpcrel_hi4)
# CHECK-INT-SAME: # encoding: [0x13,0x05,0bAAAA0101,A]
# CHECK-INT-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi4), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INT-NEXT: # <MCInst #[[#]] ADDI{{$}}
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-INT-NEXT: #  <MCOperand Reg:X10>
# CHECK-CAP-NEXT: addiy		a0, a0, %pcrel_lo(.Lpcrel_hi4)
# CHECK-CAP-SAME: # encoding: [0x1b,0x25,0bAAAA0101,A]
# CHECK-CAP-NEXT: #   fixup A - offset: 0, value: %pcrel_lo(.Lpcrel_hi4), kind: fixup_riscv_pcrel_lo12_i
# CHECK-CAP-NEXT: # <MCInst #[[#]] ADDIY{{$}}
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-CAP-NEXT: #  <MCOperand Reg:X10_Y>
# CHECK-ASM-NEXT: #  <MCOperand Expr:%pcrel_lo(.Lpcrel_hi4)>>
# CHECK-OBJ-NEXT: auipc a0, 0
# CHECK-OBJ-NEXT: R_RISCV_TLS_GD_HI20 sym
# CHECK-OBJ-NEXT: addiy a0, a0, 0
# CHECK-OBJ-NEXT: R_RISCV_PCREL_LO12_I .Lpcrel_hi4
la.tls.gd a0, sym

.data
sym:
.4byte 0
