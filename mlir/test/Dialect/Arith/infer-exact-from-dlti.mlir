// RUN: mlir-opt --split-input-file --arith-infer-exact-from-dlti %s | FileCheck %s --check-prefixes=ALL,INFER
// RUN: mlir-opt --split-input-file --arith-infer-exact-from-dlti --canonicalize %s | FileCheck %s --check-prefixes=ALL,CANON

// ALL-LABEL: func @narrowing_and_widening
// INFER: arith.index_cast %arg0 : index to i8
// INFER-NOT: exact
// INFER: arith.index_cast %arg1 exact : i8 to index
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>> } {
  func.func @narrowing_and_widening(%arg0: index, %arg1: i8) -> (i8, index) {
    %0 = arith.index_cast %arg0 : index to i8
    %1 = arith.index_cast %arg1 : i8 to index
    return %0, %1 : i8, index
  }
}

// -----

// ALL-LABEL: func @widen_to_index
// INFER: arith.index_cast %arg0 exact : i8 to index
// INFER: arith.index_castui %arg1 exact : i16 to index
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>> } {
  func.func @widen_to_index(%arg0: i8, %arg1: i16) -> (index, index) {
    %0 = arith.index_cast %arg0 : i8 to index
    %1 = arith.index_castui %arg1 : i16 to index
    return %0, %1 : index, index
  }
}

// -----

// ALL-LABEL: func @widen_to_int
// INFER: arith.index_cast %arg0 exact : index to i64
// INFER: arith.index_castui %arg0 exact : index to i64
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>> } {
  func.func @widen_to_int(%arg0: index) -> (i64, i64) {
    %0 = arith.index_cast %arg0 : index to i64
    %1 = arith.index_castui %arg0 : index to i64
    return %0, %1 : i64, i64
  }
}

// -----

// ALL-LABEL: func @roundtrip_folds
// CANON-NEXT: return %arg0 : i8
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>> } {
  func.func @roundtrip_folds(%arg0: i8) -> i8 {
    %0 = arith.index_cast %arg0 : i8 to index
    %1 = arith.index_cast %0 : index to i8
    return %1 : i8
  }
}
