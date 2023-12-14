// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @target_features
// CHECK: attributes #{{.*}} = { "target-features"="+sme,+sve,+sme-f64f64" }
llvm.func @target_features() attributes {
  target_features = #llvm.target_features<["+sme", "+sve", "+sme-f64f64"]>
} {
  llvm.return
}
