; RUN: not llc -mtriple=mips-unknown-linux -mcpu=mips32 -mattr=+use-indirect-jump-hazard %s 2>&1 | FileCheck %s

; Test that mips32 and indirect jump with hazard barriers is not supported.

; CHECK: LLVM ERROR: indirect jumps with hazard barriers requires MIPS32R2 or later
