; RUN: llc --mtriple=riscv32-unknown-linux-gnu --filetype=obj -o - %s | llvm-readelf -n - | FileCheck %s
; RUN: llc --mtriple=riscv64-unknown-linux-gnu --filetype=obj -o - %s | llvm-readelf -n - | FileCheck %s

; CHECK: Properties: RISC-V feature: ZICFISS

define i32 @f() "hw-shadow-stack" {
entry:
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-return", i32 1}
