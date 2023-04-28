// RUN: mlir-opt --arith-int-narrowing="int-bitwidths-supported=1,8,16,32" \
// RUN:          --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arith.*itofp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sitofp_extsi_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extsi_i16(%a: i16) -> f16 {
  %b = arith.extsi %a : i16 to i32
  %f = arith.sitofp %b : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @sitofp_extsi_vector_i16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : vector<3xi16> to vector<3xf16>
// CHECK-NEXT:    return %[[RET]] : vector<3xf16>
func.func @sitofp_extsi_vector_i16(%a: vector<3xi16>) -> vector<3xf16> {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %f = arith.sitofp %b : vector<3xi32> to vector<3xf16>
  return %f : vector<3xf16>
}

// CHECK-LABEL: func.func @sitofp_extsi_tensor_i16
// CHECK-SAME:    (%[[ARG:.+]]: tensor<3x?xi16>)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : tensor<3x?xi16> to tensor<3x?xf16>
// CHECK-NEXT:    return %[[RET]] : tensor<3x?xf16>
func.func @sitofp_extsi_tensor_i16(%a: tensor<3x?xi16>) -> tensor<3x?xf16> {
  %b = arith.extsi %a : tensor<3x?xi16> to tensor<3x?xi32>
  %f = arith.sitofp %b : tensor<3x?xi32> to tensor<3x?xf16>
  return %f : tensor<3x?xf16>
}

// Narrowing to i64 is not enabled in pass options.
//
// CHECK-LABEL: func.func @sitofp_extsi_i64
// CHECK-SAME:    (%[[ARG:.+]]: i64)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i64 to i128
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[EXT]] : i128 to f32
// CHECK-NEXT:    return %[[RET]] : f32
func.func @sitofp_extsi_i64(%a: i64) -> f32 {
  %b = arith.extsi %a : i64 to i128
  %f = arith.sitofp %b : i128 to f32
  return %f : f32
}

// CHECK-LABEL: func.func @uitofp_extui_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[ARG]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extui_i16(%a: i16) -> f16 {
  %b = arith.extui %a : i16 to i32
  %f = arith.uitofp %b : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @sitofp_extsi_extsi_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extsi_extsi_i8(%a: i8) -> f16 {
  %b = arith.extsi %a : i8 to i16
  %c = arith.extsi %b : i16 to i32
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_extui_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[ARG]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extui_extui_i8(%a: i8) -> f16 {
  %b = arith.extui %a : i8 to i16
  %c = arith.extui %b : i16 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_extsi_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i8 to i16
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[EXT]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extsi_extui_i8(%a: i8) -> f16 {
  %b = arith.extsi %a : i8 to i16
  %c = arith.extui %b : i16 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_trunci_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[TR:.+]]  = arith.trunci %[[ARG]] : i16 to i8
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[TR]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_trunci_extui_i8(%a: i16) -> f16 {
  %b = arith.trunci %a : i16 to i8
  %c = arith.extui %b : i8 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// This should not be folded because arith.extui changes the signed
// range of the number. For example:
//  extsi -1 : i16 to i32 ==> -1
//  extui -1 : i16 to i32 ==> U16_MAX
//
/// CHECK-LABEL: func.func @sitofp_extui_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extui %[[ARG]] : i16 to i32
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[EXT]] : i32 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extui_i16(%a: i16) -> f16 {
  %b = arith.extui %a : i16 to i32
  %f = arith.sitofp %b : i32 to f16
  return %f : f16
}

// This should not be folded because arith.extsi changes the unsigned
// range of the number. For example:
//  extsi -1 : i16 to i32 ==> U32_MAX
//  extui -1 : i16 to i32 ==> U16_MAX
//
// CHECK-LABEL: func.func @uitofp_extsi_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i16 to i32
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[EXT]] : i32 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extsi_i16(%a: i16) -> f16 {
  %b = arith.extsi %a : i16 to i32
  %f = arith.uitofp %b : i32 to f16
  return %f : f16
}

//===----------------------------------------------------------------------===//
// Commute Extension over Vector Ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extsi_over_extract_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract %[[ARG]][1] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.sitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extsi_over_extract_3xi16(%a: vector<3xi16>) -> f16 {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract %b[1] : vector<3xi32>
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extui_over_extract_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract %[[ARG]][1] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.uitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extui_over_extract_3xi16(%a: vector<3xi16>) -> f16 {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract %b[1] : vector<3xi32>
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extsi_over_extractelement_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extractelement %[[ARG]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.sitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extsi_over_extractelement_3xi16(%a: vector<3xi16>, %pos: i32) -> f16 {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extractelement %b[%pos : i32] : vector<3xi32>
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extui_over_extractelement_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extractelement %[[ARG]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.uitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extui_over_extractelement_3xi16(%a: vector<3xi16>, %pos: i32) -> f16 {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extractelement %b[%pos : i32] : vector<3xi32>
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extsi_over_extract_strided_slice_1d
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %[[ARG]] {offsets = [1], sizes = [2], strides = [1]} : vector<3xi16> to vector<2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[EXTR]] : vector<2xi16> to vector<2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2xi32>
func.func @extsi_over_extract_strided_slice_1d(%a: vector<3xi16>) -> vector<2xi32> {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1], sizes = [2], strides = [1]} : vector<3xi32> to vector<2xi32>
  return %c : vector<2xi32>
}

// CHECK-LABEL: func.func @extui_over_extract_strided_slice_1d
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %[[ARG]] {offsets = [1], sizes = [2], strides = [1]} : vector<3xi16> to vector<2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[EXTR]] : vector<2xi16> to vector<2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2xi32>
func.func @extui_over_extract_strided_slice_1d(%a: vector<3xi16>) -> vector<2xi32> {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1], sizes = [2], strides = [1]} : vector<3xi32> to vector<2xi32>
  return %c : vector<2xi32>
}

// CHECK-LABEL: func.func @extsi_over_extract_strided_slice_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi16> to vector<1x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[EXTR]] : vector<1x2xi16> to vector<1x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<1x2xi32>
func.func @extsi_over_extract_strided_slice_2d(%a: vector<2x3xi16>) -> vector<1x2xi32> {
  %b = arith.extsi %a : vector<2x3xi16> to vector<2x3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi32> to vector<1x2xi32>
  return %c : vector<1x2xi32>
}

// CHECK-LABEL: func.func @extui_over_extract_strided_slice_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi16> to vector<1x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[EXTR]] : vector<1x2xi16> to vector<1x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<1x2xi32>
func.func @extui_over_extract_strided_slice_2d(%a: vector<2x3xi16>) -> vector<1x2xi32> {
  %b = arith.extui %a : vector<2x3xi16> to vector<2x3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi32> to vector<1x2xi32>
  return %c : vector<1x2xi32>
}
