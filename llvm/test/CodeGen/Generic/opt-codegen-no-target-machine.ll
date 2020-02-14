; RUN: not --crash opt %s -dwarfehprepare -o - 2>&1 | FileCheck %s

; CHECK: Trying to construct TargetPassConfig without a target machine. Scheduling a CodeGen pass without a target triple set?
