; RUN: not opt %s -dwarf-eh-prepare -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: trying to construct TargetPassConfig without a target machine. Scheduling a CodeGen pass without a target triple set?
