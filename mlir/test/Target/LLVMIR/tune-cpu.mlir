// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @tune_cpu_x86() #[[ATTRSX86:.*]] {
// CHECK: define void @tune_cpu_arm() #[[ATTRSARM:.*]] {
// CHECK: attributes #[[ATTRSX86]] = { "tune-cpu"="pentium4" }
// CHECK: attributes #[[ATTRSARM]] = { "tune-cpu"="neoverse-n1" }

llvm.func @tune_cpu_x86() attributes {tune_cpu = "pentium4"} {
  llvm.return
}

llvm.func @tune_cpu_arm() attributes {tune_cpu = "neoverse-n1"} {
  llvm.return
}
