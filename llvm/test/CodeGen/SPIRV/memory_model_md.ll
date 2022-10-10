; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPV

; SPV: OpMemoryModel Physical32 Simple
define dso_local dllexport void @k_no_fc(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

!spirv.MemoryModel = !{!0}

!0 = !{i32 1, i32 0}
