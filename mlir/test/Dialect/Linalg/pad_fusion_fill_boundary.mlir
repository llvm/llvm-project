// RUN: mlir-opt -test-linalg-pad-fusion=fill-boundary-only -split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-linalg-pad-fusion -split-input-file %s | FileCheck %s --check-prefix=DEFAULT

func.func @fill_boundary_2d(
    %arg0 : tensor<4x3xf32>, %arg1 : f32) -> tensor<7x6xf32> {
  %init = tensor.empty() : tensor<4x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<4x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [2, 1] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<4x3xf32> to tensor<7x6xf32>
  return %1 : tensor<7x6xf32>
}

//      CHECK: func @fill_boundary_2d
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x3xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<7x6xf32>
//      CHECK:   %[[TOP_S:.+]] = tensor.extract_slice %[[EMPTY]][0, 0] [1, 6] [1, 1]
//      CHECK:   %[[TOP_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[TOP_S]]
//      CHECK:   %[[TOP:.+]] = tensor.insert_slice %[[TOP_F]] into %[[EMPTY]][0, 0] [1, 6] [1, 1]
//      CHECK:   %[[BOT_S:.+]] = tensor.extract_slice %[[TOP]][5, 0] [2, 6] [1, 1]
//      CHECK:   %[[BOT_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[BOT_S]]
//      CHECK:   %[[BOT:.+]] = tensor.insert_slice %[[BOT_F]] into %[[TOP]][5, 0] [2, 6] [1, 1]
//      CHECK:   %[[LEFT_S:.+]] = tensor.extract_slice %[[BOT]][1, 0] [4, 2] [1, 1]
//      CHECK:   %[[LEFT_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[LEFT_S]]
//      CHECK:   %[[LEFT:.+]] = tensor.insert_slice %[[LEFT_F]] into %[[BOT]][1, 0] [4, 2] [1, 1]
//      CHECK:   %[[RIGHT_S:.+]] = tensor.extract_slice %[[LEFT]][1, 5] [4, 1] [1, 1]
//      CHECK:   %[[RIGHT_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[RIGHT_S]]
//      CHECK:   %[[BOUNDARY:.+]] = tensor.insert_slice %[[RIGHT_F]] into %[[LEFT]][1, 5] [4, 1] [1, 1]
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[BOUNDARY]][1, 2] [4, 3] [1, 1]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[SLICE]] :
//      CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[BOUNDARY]][1, 2] [4, 3] [1, 1]
//      CHECK:   return %[[RESULT]]

//      DEFAULT: func @fill_boundary_2d
//      DEFAULT:   %[[EMPTY:.+]] = tensor.empty() : tensor<7x6xf32>
//      DEFAULT:   %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<7x6xf32>)
//      DEFAULT:   tensor.extract_slice %[[FILL]][1, 2] [4, 3] [1, 1]

// -----

func.func @fill_boundary_3d(%arg0 : tensor<2x2x2xf32>, %arg1 : f32) -> tensor<3x3x5xf32> {
  %init = tensor.empty() : tensor<2x2x2xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<2x2x2xf32>) outs(%init : tensor<2x2x2xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2x2xf32>
  %1 = tensor.pad %0 low [1, 0, 2] high [0, 1, 1] {
    ^bb0(%arg2: index, %arg3 : index, %arg4 : index):
      tensor.yield %arg1 : f32
    } : tensor<2x2x2xf32> to tensor<3x3x5xf32>
  return %1 : tensor<3x3x5xf32>
}

//      CHECK: func @fill_boundary_3d
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x2x2xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<3x3x5xf32>
//      CHECK:   %[[S0:.+]] = tensor.extract_slice %[[EMPTY]][0, 0, 0] [1, 3, 5] [1, 1, 1]
//      CHECK:   %[[F0:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[S0]]
//      CHECK:   %[[B0:.+]] = tensor.insert_slice %[[F0]] into %[[EMPTY]][0, 0, 0] [1, 3, 5] [1, 1, 1]
//      CHECK:   %[[S1:.+]] = tensor.extract_slice %[[B0]][1, 2, 0] [2, 1, 5] [1, 1, 1]
//      CHECK:   %[[F1:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[S1]]
//      CHECK:   %[[B1:.+]] = tensor.insert_slice %[[F1]] into %[[B0]][1, 2, 0] [2, 1, 5] [1, 1, 1]
//      CHECK:   %[[S2:.+]] = tensor.extract_slice %[[B1]][1, 0, 0] [2, 2, 2] [1, 1, 1]
//      CHECK:   %[[F2:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[S2]]
//      CHECK:   %[[B2:.+]] = tensor.insert_slice %[[F2]] into %[[B1]][1, 0, 0] [2, 2, 2] [1, 1, 1]
//      CHECK:   %[[S3:.+]] = tensor.extract_slice %[[B2]][1, 0, 4] [2, 2, 1] [1, 1, 1]
//      CHECK:   %[[F3:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[S3]]
//      CHECK:   %[[BOUNDARY:.+]] = tensor.insert_slice %[[F3]] into %[[B2]][1, 0, 4] [2, 2, 1] [1, 1, 1]
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[BOUNDARY]][1, 0, 2] [2, 2, 2] [1, 1, 1]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[SLICE]] :
//      CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[BOUNDARY]][1, 0, 2] [2, 2, 2] [1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @fill_boundary_transposed_init_map(
    %arg0 : tensor<4x3xf32>, %arg1 : f32) -> tensor<5x6xf32> {
  %init = tensor.empty() : tensor<3x4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<3x4xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<3x4xf32>
  %1 = tensor.pad %0 low [1, 2] high [1, 0] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<3x4xf32> to tensor<5x6xf32>
  return %1 : tensor<5x6xf32>
}

//      CHECK: func @fill_boundary_transposed_init_map
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x3xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<5x6xf32>
//      CHECK:   %[[TOP_S:.+]] = tensor.extract_slice %[[EMPTY]][0, 0] [1, 6] [1, 1]
//      CHECK:   %[[TOP_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[TOP_S]]
//      CHECK:   %[[TOP:.+]] = tensor.insert_slice %[[TOP_F]] into %[[EMPTY]][0, 0] [1, 6] [1, 1]
//      CHECK:   %[[BOT_S:.+]] = tensor.extract_slice %[[TOP]][4, 0] [1, 6] [1, 1]
//      CHECK:   %[[BOT_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[BOT_S]]
//      CHECK:   %[[BOT:.+]] = tensor.insert_slice %[[BOT_F]] into %[[TOP]][4, 0] [1, 6] [1, 1]
//      CHECK:   %[[LEFT_S:.+]] = tensor.extract_slice %[[BOT]][1, 0] [3, 2] [1, 1]
//      CHECK:   %[[LEFT_F:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[LEFT_S]]
//      CHECK:   %[[BOUNDARY:.+]] = tensor.insert_slice %[[LEFT_F]] into %[[BOT]][1, 0] [3, 2] [1, 1]
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[BOUNDARY]][1, 2] [3, 4] [1, 1]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[SLICE]] :
//      CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[BOUNDARY]][1, 2] [3, 4] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @no_fill_boundary_producer_reads_init(
    %arg0 : tensor<4x3xf32>, %arg1 : f32) -> tensor<7x6xf32> {
  %init = tensor.empty() : tensor<4x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    } -> tensor<4x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [2, 1] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<4x3xf32> to tensor<7x6xf32>
  return %1 : tensor<7x6xf32>
}

//      CHECK: func @no_fill_boundary_producer_reads_init
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x3xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<7x6xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[EMPTY]] : tensor<7x6xf32>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   tensor.extract_slice %[[FILL]][1, 2] [4, 3] [1, 1]

// -----

func.func @no_fill_boundary_projected_init_map(
    %arg0 : tensor<4x?xf32>, %arg1 : f32) -> tensor<6xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x?xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      linalg.yield %arg2 : f32
    } -> tensor<4xf32>
  %1 = tensor.pad %0 low [1] high [1] {
    ^bb0(%arg2: index):
      tensor.yield %arg1 : f32
    } : tensor<4xf32> to tensor<6xf32>
  return %1 : tensor<6xf32>
}

//      CHECK: func @no_fill_boundary_projected_init_map
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<6xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[EMPTY]] : tensor<6xf32>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   tensor.extract_slice %[[FILL]][1] [4] [1]

// -----

func.func @no_fill_boundary_dynamic_source(
    %arg0 : tensor<?x3xf32>, %arg1 : f32, %arg2 : index) -> tensor<?x6xf32> {
  %init = tensor.empty(%arg2) : tensor<?x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x3xf32>) outs(%init : tensor<?x3xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32):
      %1 = arith.mulf %arg3, %arg3 : f32
      linalg.yield %1 : f32
    } -> tensor<?x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [1, 1] {
    ^bb0(%arg3: index, %arg4 : index):
      tensor.yield %arg1 : f32
    } : tensor<?x3xf32> to tensor<?x6xf32>
  return %1 : tensor<?x6xf32>
}

//      CHECK: func @no_fill_boundary_dynamic_source
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x3xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?x6xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ARG1]] : f32) outs(%[[EMPTY]] : tensor<?x6xf32>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   tensor.extract_slice %[[FILL]][1, 2]

// -----

// `nofold` keeps the pad op from being folded away before the pattern runs.
func.func @fill_boundary_zero_padding(
    %arg0 : tensor<4x3xf32>, %arg1 : f32) -> tensor<4x3xf32> {
  %init = tensor.empty() : tensor<4x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<4x3xf32>
  %1 = tensor.pad %0 nofold low [0, 0] high [0, 0] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<4x3xf32> to tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}

//      CHECK: func @fill_boundary_zero_padding
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x3xf32>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[EMPTY]] :
//  CHECK-NOT:   linalg.fill
//      CHECK:   return %[[GENERIC]]

// -----

func.func @fill_boundary_second_result(
    %input : tensor<4xf32>, %init0 : tensor<4xf32>, %pad : f32) -> tensor<7xf32> {
  %init1 = tensor.empty() : tensor<4xf32>
  %results:2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%input : tensor<4xf32>) outs(%init0, %init1 : tensor<4xf32>, tensor<4xf32>) {
    ^bb0(%in : f32, %out0 : f32, %out1 : f32):
      %sum = arith.addf %in, %out0 : f32
      linalg.yield %sum, %in : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
  %padded = tensor.pad %results#1 low [1] high [2] {
    ^bb0(%i : index):
      tensor.yield %pad : f32
    } : tensor<4xf32> to tensor<7xf32>
  return %padded : tensor<7xf32>
}

// CHECK-LABEL: func @fill_boundary_second_result(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4xf32>, %[[INIT0:.*]]: tensor<4xf32>, %[[PAD:.*]]: f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7xf32>
// CHECK-NEXT: %[[LOW:.*]] = tensor.extract_slice %[[EMPTY]][0] [1] [1]
// CHECK-NEXT: %[[LOW_FILL:.*]] = linalg.fill ins(%[[PAD]] : f32) outs(%[[LOW]]
// CHECK-NEXT: %[[LOW_INSERT:.*]] = tensor.insert_slice %[[LOW_FILL]] into %[[EMPTY]][0] [1] [1]
// CHECK-NEXT: %[[HIGH:.*]] = tensor.extract_slice %[[LOW_INSERT]][5] [2] [1]
// CHECK-NEXT: %[[HIGH_FILL:.*]] = linalg.fill ins(%[[PAD]] : f32) outs(%[[HIGH]]
// CHECK-NEXT: %[[BOUNDARY:.*]] = tensor.insert_slice %[[HIGH_FILL]] into %[[LOW_INSERT]][5] [2] [1]
// CHECK-NEXT: %[[INTERIOR:.*]] = tensor.extract_slice %[[BOUNDARY]][1] [4] [1]
// CHECK-NEXT: %[[RESULTS:.*]]:2 = linalg.generic
// CHECK-SAME: outs(%[[INIT0]], %[[INTERIOR]] : tensor<4xf32>, tensor<4xf32>)
// CHECK: %[[RESULT:.*]] = tensor.insert_slice %[[RESULTS]]#1 into %[[BOUNDARY]][1] [4] [1]
// CHECK-NEXT: return %[[RESULT]]


// -----

func.func @fill_boundary_empty_source_first_dim(
    %arg0 : tensor<0x3xf32>, %arg1 : f32) -> tensor<3x6xf32> {
  %init = tensor.empty() : tensor<0x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<0x3xf32>) outs(%init : tensor<0x3xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<0x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [2, 1] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<0x3xf32> to tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

//      CHECK: func @fill_boundary_empty_source_first_dim
//      CHECK:   tensor.empty() : tensor<3x6xf32>
//      CHECK:   tensor.extract_slice %{{.+}}[0, 0] [1, 6] [1, 1]
//      CHECK:   linalg.fill
//      CHECK:   tensor.extract_slice %{{.+}}[1, 0] [2, 6] [1, 1]
//      CHECK:   linalg.fill
//  CHECK-NOT:   linalg.fill
//      CHECK:   return

// -----

func.func @fill_boundary_empty_source_later_dim(
    %arg0 : tensor<3x0xf32>, %arg1 : f32) -> tensor<6x3xf32> {
  %init = tensor.empty() : tensor<3x0xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<3x0xf32>) outs(%init : tensor<3x0xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<3x0xf32>
  %1 = tensor.pad %0 low [1, 2] high [2, 1] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<3x0xf32> to tensor<6x3xf32>
  return %1 : tensor<6x3xf32>
}

//      CHECK: func @fill_boundary_empty_source_later_dim
//      CHECK:   tensor.empty() : tensor<6x3xf32>
//      CHECK:   tensor.extract_slice %{{.+}}[0, 0] [1, 3] [1, 1]
//      CHECK:   linalg.fill
//      CHECK:   tensor.extract_slice %{{.+}}[4, 0] [2, 3] [1, 1]
//      CHECK:   linalg.fill
//      CHECK:   tensor.extract_slice %{{.+}}[1, 0] [3, 2] [1, 1]
//      CHECK:   linalg.fill
//      CHECK:   tensor.extract_slice %{{.+}}[1, 2] [3, 1] [1, 1]
//      CHECK:   linalg.fill
//  CHECK-NOT:   linalg.fill
//      CHECK:   return

// -----

func.func @fill_boundary_dynamic_high_pad(
    %arg0 : tensor<4x3xf32>, %arg1 : f32, %arg2 : index) -> tensor<7x6xf32> {
  %init = tensor.empty() : tensor<4x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32):
      %1 = arith.mulf %arg3, %arg3 : f32
      linalg.yield %1 : f32
    } -> tensor<4x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [%arg2, 1] {
    ^bb0(%arg3: index, %arg4 : index):
      tensor.yield %arg1 : f32
    } : tensor<4x3xf32> to tensor<7x6xf32>
  return %1 : tensor<7x6xf32>
}

//      CHECK: func @fill_boundary_dynamic_high_pad
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<7x6xf32>
//      CHECK:   tensor.extract_slice %[[EMPTY]][0, 0] [1, 6] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32)
//      CHECK:   tensor.extract_slice %{{.+}}[5, 0] [2, 6] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32)
//      CHECK:   tensor.extract_slice %{{.+}}[1, 0] [4, 2] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32)
//      CHECK:   tensor.extract_slice %{{.+}}[1, 5] [4, 1] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32)
//  CHECK-NOT:   linalg.fill
//      CHECK:   tensor.extract_slice %{{.+}}[1, 2] [4, 3] [1, 1]
//      CHECK:   linalg.generic
