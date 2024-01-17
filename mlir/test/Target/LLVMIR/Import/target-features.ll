; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @target_features()
; CHECK-SAME: #llvm.target_features<["+sme", "+sme-f64f64", "+sve"]>
define void @target_features() #0 {
  ret void
}

attributes #0 = { "target-features"="+sme,+sme-f64f64,+sve" }
