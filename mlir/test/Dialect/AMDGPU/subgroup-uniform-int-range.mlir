// RUN: mlir-opt --arith-int-range-narrowing="int-bitwidths-supported=32" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @narrow
//       CHECK:   %[[SRC:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:   %[[CAST1:.*]] = arith.index_castui %[[SRC]] : index to i32
//       CHECK:   %[[VAL:.*]] = amdgpu.assume_subgroup_uniform %[[CAST1]] : i32
//       CHECK:   %[[CAST2:.*]] = arith.index_castui %[[VAL]] : i32 to index
//       CHECK:   return %[[CAST2]] : index
func.func @narrow() -> index {
  %0 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %1 = amdgpu.assume_subgroup_uniform %0 : index
  return %1: index
}
