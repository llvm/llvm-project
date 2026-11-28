// RUN: mlir-opt %s -convert-to-llvm="filter-dialects=func add-lowering=arm_sve" | FileCheck %s
// RUN: not mlir-opt %s -convert-to-llvm="add-lowering=tensor" 2>&1 | FileCheck %s --check-prefix=NO-IFACE

// The dialects of `add-lowering` are loaded even if the input contains no
// operation of them, and their patterns extend `filter-dialects`.

// CHECK-LABEL: llvm.func @create_mask(
// CHECK:         "arm_sve.intr.whilelt"
// CHECK-NOT:     vector.create_mask

// NO-IFACE: error: dialect does not implement ConvertToLLVMPatternInterface: tensor
func.func @create_mask(%index: index) -> vector<[4]xi1> {
  %0 = vector.create_mask %index : vector<[4]xi1>
  return %0 : vector<[4]xi1>
}
