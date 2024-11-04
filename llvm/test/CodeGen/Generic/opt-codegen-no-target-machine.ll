; RUN: not --crash opt %s -dwarf-eh-prepare -o - 2>&1 | FileCheck %s

; CHECK: Trying to construct TargetPassConfig without a target machine. Scheduling a CodeGen pass without a target triple set?
