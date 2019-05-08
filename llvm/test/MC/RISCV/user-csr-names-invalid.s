# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:  | FileCheck -check-prefixes=CHECK-NEED-RV32 %s

# These user mode CSR register names are RV32 only.

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
