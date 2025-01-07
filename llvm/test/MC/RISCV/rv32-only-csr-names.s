# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK-NEED-RV32 %s

# The following CSR register names are all RV32 only.

csrrs t1, cycleh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'cycleh' is RV32 only
csrrs t1, timeh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'timeh' is RV32 only
csrrs t1, instreth, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'instreth' is RV32 only

csrrs t1, hpmcounter3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter3h' is RV32 only
csrrs t1, hpmcounter4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter4h' is RV32 only
csrrs t1, hpmcounter5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter5h' is RV32 only
csrrs t1, hpmcounter6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter6h' is RV32 only
csrrs t1, hpmcounter7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter7h' is RV32 only
csrrs t1, hpmcounter8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter8h' is RV32 only
csrrs t1, hpmcounter9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter9h' is RV32 only
csrrs t1, hpmcounter10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter10h' is RV32 only
csrrs t1, hpmcounter11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter11h' is RV32 only
csrrs t1, hpmcounter12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter12h' is RV32 only
csrrs t1, hpmcounter13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter13h' is RV32 only
csrrs t1, hpmcounter14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter14h' is RV32 only
csrrs t1, hpmcounter15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter15h' is RV32 only
csrrs t1, hpmcounter16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter16h' is RV32 only
csrrs t1, hpmcounter17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter17h' is RV32 only
csrrs t1, hpmcounter18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter18h' is RV32 only
csrrs t1, hpmcounter19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter19h' is RV32 only
csrrs t1, hpmcounter20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter20h' is RV32 only
csrrs t1, hpmcounter21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter21h' is RV32 only
csrrs t1, hpmcounter22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter22h' is RV32 only
csrrs t1, hpmcounter23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter23h' is RV32 only
csrrs t1, hpmcounter24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter24h' is RV32 only
csrrs t1, hpmcounter25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter25h' is RV32 only
csrrs t1, hpmcounter26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter26h' is RV32 only
csrrs t1, hpmcounter27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter27h' is RV32 only
csrrs t1, hpmcounter28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter28h' is RV32 only
csrrs t1, hpmcounter29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter29h' is RV32 only
csrrs t1, hpmcounter30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter30h' is RV32 only
csrrs t1, hpmcounter31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hpmcounter31h' is RV32 only

csrrs t1, henvcfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'henvcfgh' is RV32 only

csrrs t1, htimedeltah, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'htimedeltah' is RV32 only

csrrs t1, mstatush, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mstatush' is RV32 only

csrrs t1, menvcfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'menvcfgh' is RV32 only

csrrs t1, mseccfgh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mseccfgh' is RV32 only

csrrs t1, pmpcfg1, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg1' is RV32 only
csrrs t1, pmpcfg3, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg3' is RV32 only
csrrs t1, pmpcfg5, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg5' is RV32 only
csrrs t1, pmpcfg7, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg7' is RV32 only
csrrs t1, pmpcfg9, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg9' is RV32 only
csrrs t1, pmpcfg11, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg11' is RV32 only
csrrs t1, pmpcfg13, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg13' is RV32 only
csrrs t1, pmpcfg15, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'pmpcfg15' is RV32 only

csrrs t1, mcycleh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mcycleh' is RV32 only
csrrs t1, minstreth, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'minstreth' is RV32 only

csrrs t1, mhpmcounter3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter3h' is RV32 only
csrrs t1, mhpmcounter4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter4h' is RV32 only
csrrs t1, mhpmcounter5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter5h' is RV32 only
csrrs t1, mhpmcounter6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter6h' is RV32 only
csrrs t1, mhpmcounter7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter7h' is RV32 only
csrrs t1, mhpmcounter8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter8h' is RV32 only
csrrs t1, mhpmcounter9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter9h' is RV32 only
csrrs t1, mhpmcounter10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter10h' is RV32 only
csrrs t1, mhpmcounter11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter11h' is RV32 only
csrrs t1, mhpmcounter12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter12h' is RV32 only
csrrs t1, mhpmcounter13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter13h' is RV32 only
csrrs t1, mhpmcounter14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter14h' is RV32 only
csrrs t1, mhpmcounter15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter15h' is RV32 only
csrrs t1, mhpmcounter16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter16h' is RV32 only
csrrs t1, mhpmcounter17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter17h' is RV32 only
csrrs t1, mhpmcounter18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter18h' is RV32 only
csrrs t1, mhpmcounter19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter19h' is RV32 only
csrrs t1, mhpmcounter20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter20h' is RV32 only
csrrs t1, mhpmcounter21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter21h' is RV32 only
csrrs t1, mhpmcounter22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter22h' is RV32 only
csrrs t1, mhpmcounter23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter23h' is RV32 only
csrrs t1, mhpmcounter24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter24h' is RV32 only
csrrs t1, mhpmcounter25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter25h' is RV32 only
csrrs t1, mhpmcounter26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter26h' is RV32 only
csrrs t1, mhpmcounter27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter27h' is RV32 only
csrrs t1, mhpmcounter28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter28h' is RV32 only
csrrs t1, mhpmcounter29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter29h' is RV32 only
csrrs t1, mhpmcounter30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter30h' is RV32 only
csrrs t1, mhpmcounter31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmcounter31h' is RV32 only

csrrs t1, mhpmevent3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent3h' is RV32 only
csrrs t1, mhpmevent4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent4h' is RV32 only
csrrs t1, mhpmevent5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent5h' is RV32 only
csrrs t1, mhpmevent6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent6h' is RV32 only
csrrs t1, mhpmevent7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent7h' is RV32 only
csrrs t1, mhpmevent8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent8h' is RV32 only
csrrs t1, mhpmevent9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent9h' is RV32 only
csrrs t1, mhpmevent10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent10h' is RV32 only
csrrs t1, mhpmevent11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent11h' is RV32 only
csrrs t1, mhpmevent12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent12h' is RV32 only
csrrs t1, mhpmevent13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent13h' is RV32 only
csrrs t1, mhpmevent14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent14h' is RV32 only
csrrs t1, mhpmevent15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent15h' is RV32 only
csrrs t1, mhpmevent16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent16h' is RV32 only
csrrs t1, mhpmevent17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent17h' is RV32 only
csrrs t1, mhpmevent18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent18h' is RV32 only
csrrs t1, mhpmevent19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent19h' is RV32 only
csrrs t1, mhpmevent20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent20h' is RV32 only
csrrs t1, mhpmevent21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent21h' is RV32 only
csrrs t1, mhpmevent22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent22h' is RV32 only
csrrs t1, mhpmevent23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent23h' is RV32 only
csrrs t1, mhpmevent24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent24h' is RV32 only
csrrs t1, mhpmevent25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent25h' is RV32 only
csrrs t1, mhpmevent26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent26h' is RV32 only
csrrs t1, mhpmevent27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent27h' is RV32 only
csrrs t1, mhpmevent28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent28h' is RV32 only
csrrs t1, mhpmevent29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent29h' is RV32 only
csrrs t1, mhpmevent30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent30h' is RV32 only
csrrs t1, mhpmevent31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mhpmevent31h' is RV32 only

csrrs t1, mstateen0h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mstateen0h' is RV32 only
csrrs t1, mstateen1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mstateen1h' is RV32 only
csrrs t1, mstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mstateen3h' is RV32 only
csrrs t1, mstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mstateen3h' is RV32 only

csrrs t1, hstateen0h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hstateen0h' is RV32 only
csrrs t1, hstateen1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hstateen1h' is RV32 only
csrrs t1, hstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hstateen3h' is RV32 only
csrrs t1, hstateen3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hstateen3h' is RV32 only

csrrs t1, stimecmph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'stimecmph' is RV32 only
csrrs t1, vstimecmph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'vstimecmph' is RV32 only

csrrs t1, midelegh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'midelegh' is RV32 only
csrrs t1, mieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mieh' is RV32 only
csrrs t1, mvienh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mvienh' is RV32 only
csrrs t1, mviph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'mviph' is RV32 only
csrrs t1, miph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'miph' is RV32 only
csrrs t1, sieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'sieh' is RV32 only
csrrs t1, siph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'siph' is RV32 only
csrrs t1, hidelegh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hidelegh' is RV32 only
csrrs t1, hvienh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hvienh' is RV32 only
csrrs t1, hviph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hviph' is RV32 only
csrrs t1, hviprio1h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hviprio1h' is RV32 only
csrrs t1, hviprio2h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'hviprio2h' is RV32 only
csrrs t1, vsieh, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'vsieh' is RV32 only
csrrs t1, vsiph, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register 'vsiph' is RV32 only
