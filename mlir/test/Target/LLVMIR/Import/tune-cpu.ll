; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @tune_cpu_x86()
; CHECK-SAME: tune_cpu = "pentium4"
define void @tune_cpu_x86() #0 {
  ret void
}

; CHECK-LABEL: llvm.func @tune_cpu_arm()
; CHECK-SAME: tune_cpu = "neoverse-n1"
define void @tune_cpu_arm() #1 {
  ret void
}

attributes #0 = { "tune-cpu"="pentium4" }
attributes #1 = { "tune-cpu"="neoverse-n1" }
