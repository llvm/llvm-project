# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK-NEED-RV32 %s

# These machine mode CSR register names are RV32 only.

csrrs t1, pmpcfg1, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, pmpcfg3, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled

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
