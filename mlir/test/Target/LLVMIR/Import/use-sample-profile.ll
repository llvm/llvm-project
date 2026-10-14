; RUN: mlir-translate -import-llvm  %s | FileCheck %s

; CHECK-LABEL: llvm.func @use_sample_profile()
; CHECK-SAME: use_sample_profile = true
define void @use_sample_profile() #0 {
  ret void
}

attributes #0 = { "use-sample-profile" }
