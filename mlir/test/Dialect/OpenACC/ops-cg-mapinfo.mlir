// RUN: mlir-opt -split-input-file %s | FileCheck %s --check-prefixes=CHECK
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK

// -----

// The whole object is mapped, so its byte size is stated on the operation and
// no descriptor is involved.
// CHECK-LABEL: func @map_info_whole_object
func.func @map_info_whole_object(%a: memref<10xf32>) {
  %size = arith.constant 40 : i64
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      size(%size : i64) elementSize(4) name("a")
      descKind(none) mapFlags(to, from) -> memref<10xf32>
  acc.data dataOperands(%map : memref<10xf32>) {
    acc.terminator
  }
  return
}
// CHECK: %[[SIZE:.*]] = arith.constant 40 : i64
// CHECK: acc.map_info varPtr(%{{.*}} : memref<10xf32>) varType(tensor<10xf32>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: name("a")
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to,from)
// CHECK-SAME: -> memref<10xf32>

// -----

// A partial array map carries bounds instead of a byte size, and the source
// extent records how large the whole array is so a strided transfer can be
// described.
// CHECK-LABEL: func @map_info_bounds
func.func @map_info_bounds(%a: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %size = arith.constant 0 : i64
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c4 : index) stride(%c1 : index) startIdx(%c0 : index)
      sourceExtent(%c10 : index)
  %map = acc.map_info varPtr(%a : memref<?xf32>) varType(tensor<?xf32>)
      bounds(%bounds) size(%size : i64) elementSize(4)
      descKind(openacc) mapFlags(to) -> memref<?xf32>
  acc.data dataOperands(%map : memref<?xf32>) {
    acc.terminator
  }
  return
}
// CHECK: %[[BOUNDS:.*]] = acc.bounds
// CHECK-SAME: sourceExtent(%{{[^)]*}})
// CHECK: acc.map_info varPtr(%{{.*}} : memref<?xf32>) varType(tensor<?xf32>)
// CHECK-SAME: bounds(%[[BOUNDS]])
// CHECK-SAME: descKind(openacc)
// CHECK-SAME: mapFlags(to)

// -----

// A descriptor-backed map names the attach point and the descriptor that holds
// it, and defers the mapped size to that descriptor.
// CHECK-LABEL: func @map_info_descriptor
func.func @map_info_descriptor(%a: memref<10xf32>, %slot: memref<i64>,
    %desc: memref<i64>) {
  %size = arith.constant 0 : i64
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      varPtrPtr(%slot : memref<i64>) desc(%desc : memref<i64>)
      size(%size : i64) name("p") exitLoc(loc("p.f90":7:3))
      descKind(cfi) mapFlags(to, from, ptr_and_obj) -> memref<10xf32>
  acc.data dataOperands(%map : memref<10xf32>) {
    acc.terminator
  }
  return
}
// CHECK: acc.map_info varPtr(%{{[^)]*}}) varType(tensor<10xf32>)
// CHECK-SAME: varPtrPtr(%{{[^)]*}})
// CHECK-SAME: desc(%{{[^)]*}})
// CHECK-SAME: exitLoc(
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,from,ptr_and_obj)

// -----

// A nested descriptor names every layout that describes the object.
// CHECK-LABEL: func @map_info_nested_desc_kind
func.func @map_info_nested_desc_kind(%a: memref<?xf32>, %desc: memref<i64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c4 : index) stride(%c1 : index) startIdx(%c0 : index)
  %map = acc.map_info varPtr(%a : memref<?xf32>) varType(tensor<?xf32>)
      desc(%desc : memref<i64>) bounds(%bounds) elementSize(4)
      descKind(cfi, openacc) mapFlags(to) -> memref<?xf32>
  acc.data dataOperands(%map : memref<?xf32>) {
    acc.terminator
  }
  return
}
// CHECK: acc.map_info
// CHECK-SAME: descKind(cfi,openacc)

// -----

// A size that is not known at compile time is stated as -1, and a size that is
// only known at run time is supplied as a value.
// CHECK-LABEL: func @map_info_unknown_and_runtime_size
func.func @map_info_unknown_and_runtime_size(%a: memref<10xf32>,
    %runtime: i64) {
  %unknown = arith.constant -1 : i64
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      size(%unknown : i64) descKind(none) mapFlags(to) -> memref<10xf32>
  %dynamic = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      size(%runtime : i64) descKind(none) mapFlags(to) -> memref<10xf32>
  acc.data dataOperands(%map, %dynamic : memref<10xf32>, memref<10xf32>) {
    acc.terminator
  }
  return
}
// CHECK: %[[UNKNOWN:.*]] = arith.constant -1 : i64
// CHECK: acc.map_info
// CHECK-SAME: size(%[[UNKNOWN]] : i64)
// CHECK: acc.map_info
// CHECK-SAME: size(%{{[^)]*}})

// -----

// A map with no flags describes an entry whose transfer behavior is decided by
// the construct that uses it.
// CHECK-LABEL: func @map_info_no_flags
func.func @map_info_no_flags(%a: memref<10xf32>) {
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      descKind(none) mapFlags(none) -> memref<10xf32>
  %token = acc.declare_enter dataOperands(%map : memref<10xf32>)
  acc.declare_exit token(%token) dataOperands(%map : memref<10xf32>)
  return
}
// CHECK: acc.map_info varPtr(%{{[^)]*}}) varType(tensor<10xf32>)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(none)
