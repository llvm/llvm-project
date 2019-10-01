; RUN: not llc -mtriple=riscv64 -mattr=+e < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV64E %s

; RV64E: LLVM ERROR: RV32E can't be enabled for an RV64 target
