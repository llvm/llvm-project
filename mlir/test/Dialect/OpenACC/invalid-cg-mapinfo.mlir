// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

func.func @map_info_var_type_is_pointer(%a: memref<10xf32>) {
  // expected-error@+1 {{'acc.map_info' op varType must capture the element type of var}}
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(memref<10xf32>)
      descKind(none) mapFlags(to) -> memref<10xf32>
  return
}

// -----

func.func @map_info_desc_without_kind(%a: memref<10xf32>, %desc: memref<i64>) {
  // expected-error@+1 {{'acc.map_info' op desc requires a descKind other than none}}
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      desc(%desc : memref<i64>) descKind(none) mapFlags(to) -> memref<10xf32>
  return
}

// -----

func.func @map_info_bounds_without_openacc_kind(%a: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c4 : index) stride(%c1 : index) startIdx(%c0 : index)
  // expected-error@+1 {{'acc.map_info' op bounds require descKind openacc}}
  %map = acc.map_info varPtr(%a : memref<?xf32>) varType(tensor<?xf32>)
      bounds(%bounds) elementSize(4) descKind(cfi) mapFlags(to)
      -> memref<?xf32>
  return
}

// -----

func.func @map_info_negative_size(%a: memref<10xf32>) {
  %size = arith.constant -2 : i64
  // expected-error@+1 {{'acc.map_info' op size must be -1, 0, or a positive byte count}}
  %map = acc.map_info varPtr(%a : memref<10xf32>) varType(tensor<10xf32>)
      size(%size : i64) descKind(none) mapFlags(to) -> memref<10xf32>
  return
}
