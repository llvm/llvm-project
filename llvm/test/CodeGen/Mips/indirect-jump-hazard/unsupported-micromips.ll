; RUN: not llc -mtriple=mips-unknown-linux -mcpu=mips32r2 -mattr=+micromips,+use-indirect-jump-hazard %s 2>&1 | FileCheck %s

; Test that microMIPS and indirect jump with hazard barriers is not supported.

; CHECK: LLVM ERROR: cannot combine indirect jumps with hazard barriers and microMIPS
