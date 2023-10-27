# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK-NEED-RV32 %s

# The following CSR register names are all RV32 only.

csrrs t1, cycleh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, timeh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, instreth, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, hpmcounter3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hpmcounter31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, henvcfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, htimedeltah, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mstatush, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, menvcfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mseccfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, pmpcfg1, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg3, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg5, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg7, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg9, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg11, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg13, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg15, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mcycleh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, minstreth, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mhpmcounter3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmcounter31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mhpmevent3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, mstateen0h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mstateen1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, hstateen0h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hstateen1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, stimecmph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, vstimecmph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

csrrs t1, midelegh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mvienh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mviph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, miph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, sieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, siph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hidelegh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hvienh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hviph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hviprio1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, hviprio2h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, vsieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, vsiph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
