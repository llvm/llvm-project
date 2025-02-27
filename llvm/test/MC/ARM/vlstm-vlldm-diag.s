// RUN: not llvm-mc -triple=armv8.1m.main-arm-none-eabi -mcpu=generic -show-encoding %s 2>&1 >/dev/null \
// RUN: | FileCheck --check-prefixes=ERR %s

// RUN: not llvm-mc -triple=armv8.1m.main-arm-none-eabi -mcpu=generic -show-encoding %s 2>&1 >/dev/null \
// RUN: | FileCheck --check-prefixes=ERRT2 %s

vlstm r8, {d0 - d11}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlstm r8, {d0 - d11}

vlldm r8, {d0 - d11}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlldm r8, {d0 - d11}

vlstm r8, {d3 - d15}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlstm r8, {d3 - d15}

vlldm r8, {d3 - d15}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlldm r8, {d3 - d15}

vlstm r8, {d0 - d29}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlstm r8, {d0 - d29}

vlldm r8, {d0 - d29}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlldm r8, {d0 - d29}

vlstm r8, {d3 - d31}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlstm r8, {d3 - d31}

vlldm r8, {d3 - d31}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlldm r8, {d3 - d31}

vlstm r8, {d31}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlstm r8, {d31}

vlldm r8, {d31}
// ERR: error: operand must be exactly {d0-d15} (T1) or {d0-d31} (T2)
// ERR-NEXT: vlldm r8, {d31}

vlstm r8, {d0 - d35}
// ERR: error: register expected
// ERR-NEXT: vlstm r8, {d0 - d35}

vlldm r8, {d0 - d35}
// ERR: error: register expected
// ERR-NEXT: vlldm r8, {d0 - d35}

vlstm pc
// ERR: error: operand must be a register in range [r0, r14]
// ERR-NEXT: vlstm pc

vlldm pc
// ERR: error: operand must be a register in range [r0, r14]
// ERR-NEXT: vlldm pc

vlstm pc
// ERRT2: error: operand must be a register in range [r0, r14]
// ERRT2-NEXT: vlstm pc

vlldm pc
// ERRT2: error: operand must be a register in range [r0, r14]
// ERRT2-NEXT: vlldm pc