// RUN: mlir-opt --arith-int-narrowing="int-bitwidths-supported=1,8,16,32" \
// RUN:          --verify-diagnostics %s | FileCheck %s

// Check that we can calculate `linalg.index` value bounds and use them to
// optimize index casts.

//===----------------------------------------------------------------------===//
// arith.index_cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @linalg_indexcast_dim_0_i8
// CHECK:         %[[IDX:.+]] = linalg.index 0 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_cast %[[IDX]] : index to i8
// CHECK-NEXT:    %[[FP:.+]]  = arith.sitofp %[[INT]] : i8 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcast_dim_0_i8(%arg0: tensor<f16>) -> tensor<128xf16> {
  %init = tensor.empty() : tensor<128xf16>
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%init : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 0 : index
    %int = arith.index_cast %idx : index to i64
    %fp = arith.sitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<128xf16>

  return %res : tensor<128xf16>
}

// CHECK-LABEL: func @linalg_indexcast_dim_1_i16
// CHECK:         %[[IDX:.+]] = linalg.index 1 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_cast %[[IDX]] : index to i16
// CHECK-NEXT:    %[[FP:.+]]  = arith.sitofp %[[INT]] : i16 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcast_dim_1_i16(%arg0: tensor<f16>, %arg1: tensor<?x129xf16>) -> tensor<?x129xf16> {
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%arg1 : tensor<?x129xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 1 : index
    %int = arith.index_cast %idx : index to i64
    %fp = arith.sitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<?x129xf16>

  return %res : tensor<?x129xf16>
}

// CHECK-LABEL: func @linalg_indexcast_dynamic_dim_i64
// CHECK:         %[[IDX:.+]] = linalg.index 0 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_cast %[[IDX]] : index to i64
// CHECK-NEXT:    %[[FP:.+]]  = arith.sitofp %[[INT]] : i64 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcast_dynamic_dim_i64(%arg0: tensor<f16>, %arg1: tensor<?xf16>) -> tensor<?xf16> {
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%arg1 : tensor<?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 0 : index
    %int = arith.index_cast %idx : index to i64
    %fp = arith.sitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<?xf16>

  return %res : tensor<?xf16>
}

//===----------------------------------------------------------------------===//
// arith.index_castui
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @linalg_indexcastui_dim_0_i8
// CHECK:         %[[IDX:.+]] = linalg.index 0 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_castui %[[IDX]] : index to i8
// CHECK-NEXT:    %[[FP:.+]]  = arith.uitofp %[[INT]] : i8 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcastui_dim_0_i8(%arg0: tensor<f16>) -> tensor<256xf16> {
  %init = tensor.empty() : tensor<256xf16>
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%init : tensor<256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 0 : index
    %int = arith.index_castui %idx : index to i64
    %fp = arith.uitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<256xf16>

  return %res : tensor<256xf16>
}

// CHECK-LABEL: func @linalg_indexcastui_dim_1_i16
// CHECK:         %[[IDX:.+]] = linalg.index 1 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_castui %[[IDX]] : index to i16
// CHECK-NEXT:    %[[FP:.+]]  = arith.uitofp %[[INT]] : i16 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcastui_dim_1_i16(%arg0: tensor<f16>, %arg1: tensor<?x257xf16>) -> tensor<?x257xf16> {
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%arg1 : tensor<?x257xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 1 : index
    %int = arith.index_castui %idx : index to i64
    %fp = arith.uitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<?x257xf16>

  return %res : tensor<?x257xf16>
}

// CHECK-LABEL: func @linalg_indexcastui_dynamic_dim_i64
// CHECK:         %[[IDX:.+]] = linalg.index 0 : index
// CHECK-NEXT:    %[[INT:.+]] = arith.index_castui %[[IDX]] : index to i64
// CHECK-NEXT:    %[[FP:.+]]  = arith.uitofp %[[INT]] : i64 to f16
// CHECK-NEXT:    linalg.yield %[[FP]] : f16
func.func @linalg_indexcastui_dynamic_dim_i64(%arg0: tensor<f16>, %arg1: tensor<?xf16>) -> tensor<?xf16> {
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%arg0 : tensor<f16>)
    outs(%arg1 : tensor<?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %idx = linalg.index 0 : index
    %int = arith.index_castui %idx : index to i64
    %fp = arith.uitofp %int : i64 to f16
    linalg.yield %fp : f16
  } -> tensor<?xf16>

  return %res : tensor<?xf16>
}
