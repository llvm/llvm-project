; RUN: not --crash llc -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Unknown memory model in spirv.MemoryModel metadata
define void @main() {
entry:
  ret void
}

; AddressingModel=Logical (0), MemoryModel=Unknown (99)
!spirv.MemoryModel = !{!0}
!0 = !{i32 0, i32 99}
