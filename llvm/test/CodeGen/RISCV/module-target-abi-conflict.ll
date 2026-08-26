; A "target-abi" module flag that conflicts with the -target-abi
; command-line option is an error, which should be diagnosed exactly
; once

; RUN: not llc -target-abi=lp64d -filetype=null < %s 2>&1 | FileCheck %s -implicit-check-not=error:
; RUN: not llc -enable-new-pm -target-abi=lp64d -filetype=null < %s 2>&1 | FileCheck %s -implicit-check-not=error:

; CHECK: error: -target-abi option != target-abi module flag

target triple = "riscv64"

define void @f1() #0 {
  ret void
}

define void @f2() #1 {
  ret void
}

attributes #0 = { "target-cpu"="generic-rv64" }
attributes #1 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64"}
