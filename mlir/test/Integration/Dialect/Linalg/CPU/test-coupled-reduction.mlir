// RUN: mlir-opt %s --transform-preload-library="transform-library-paths=%S/Inputs/coupled-reduction.mlir" --transform-interpreter="entry-point=split_forall" --canonicalize --cse --one-shot-bufferize=bufferize-function-boundaries --scf-forall-to-for --convert-linalg-to-loops --buffer-deallocation-pipeline --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 | FileCheck %s
// RUN: mlir-opt %s --transform-preload-library="transform-library-paths=%S/Inputs/coupled-reduction.mlir" --transform-interpreter="entry-point=split_forall" --canonicalize --cse --one-shot-bufferize=bufferize-function-boundaries --scf-forall-to-for --convert-linalg-to-loops --buffer-deallocation-pipeline --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O2 -e main -entry-point-result=i32 | FileCheck %s
// RUN: mlir-opt %s --transform-preload-library="transform-library-paths=%S/Inputs/coupled-reduction.mlir" --transform-interpreter="entry-point=split_for" --canonicalize --cse --one-shot-bufferize=bufferize-function-boundaries --scf-forall-to-for --convert-linalg-to-loops --buffer-deallocation-pipeline --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 | FileCheck %s
// RUN: mlir-opt %s --transform-preload-library="transform-library-paths=%S/Inputs/coupled-reduction.mlir" --transform-interpreter="entry-point=split_for" --canonicalize --cse --one-shot-bufferize=bufferize-function-boundaries --scf-forall-to-for --convert-linalg-to-loops --buffer-deallocation-pipeline --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-index-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O2 -e main -entry-point-result=i32 | FileCheck %s

// CHECK: {{^0$}}
// Stable cross-partition ties, a winning non-neutral seed, and a non-adjacent
// value/index pair alongside an independent sum initialized to seven.
// The dynamic empty reduction preserves its original initial pair.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @signed_max(%arg0: tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %0 = tensor.empty() : tensor<3xi32>
    %1 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%0 : tensor<3xi32>) -> tensor<3xi32>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.fill ins(%c9223372036854775807_i64 : i64) outs(%2 : tensor<3xi64>) -> tensor<3xi64>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<9x3xi32>) outs(%1, %3 : tensor<3xi32>, tensor<3xi64>) attrs =  {test.coupled_pair} {
    ^bb0(%in: i32, %out: i32, %out_0: i64):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maxsi %in, %out : i32
      %8 = arith.cmpi sgt, %in, %out : i32
      %9 = arith.cmpi eq, %in, %out : i32
      %10 = arith.cmpi slt, %6, %out_0 : i64
      %11 = arith.andi %9, %10 : i1
      %12 = arith.ori %8, %11 : i1
      %13 = arith.select %12, %6, %out_0 : i64
      linalg.yield %7, %13 : i32, i64
    } -> (tensor<3xi32>, tensor<3xi64>)
    return %4#0, %4#1 : tensor<3xi32>, tensor<3xi64>
  }
  func.func @winning_seed(%arg0: tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>) {
    %c200_i32 = arith.constant 200 : i32
    %c-7_i64 = arith.constant -7 : i64
    %0 = tensor.empty() : tensor<3xi32>
    %1 = linalg.fill ins(%c200_i32 : i32) outs(%0 : tensor<3xi32>) -> tensor<3xi32>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.fill ins(%c-7_i64 : i64) outs(%2 : tensor<3xi64>) -> tensor<3xi64>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<9x3xi32>) outs(%1, %3 : tensor<3xi32>, tensor<3xi64>) attrs =  {test.coupled_pair} {
    ^bb0(%in: i32, %out: i32, %out_0: i64):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maxsi %in, %out : i32
      %8 = arith.cmpi sgt, %in, %out : i32
      %9 = arith.cmpi eq, %in, %out : i32
      %10 = arith.cmpi slt, %6, %out_0 : i64
      %11 = arith.andi %9, %10 : i1
      %12 = arith.ori %8, %11 : i1
      %13 = arith.select %12, %6, %out_0 : i64
      linalg.yield %7, %13 : i32, i64
    } -> (tensor<3xi32>, tensor<3xi64>)
    return %4#0, %4#1 : tensor<3xi32>, tensor<3xi64>
  }
  func.func @mixed_sum(%arg0: tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>, tensor<3xi64>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %0 = tensor.empty() : tensor<3xi32>
    %1 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%0 : tensor<3xi32>) -> tensor<3xi32>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.fill ins(%c9223372036854775807_i64 : i64) outs(%2 : tensor<3xi64>) -> tensor<3xi64>
    %c7_i64 = arith.constant 7 : i64
    %4 = tensor.empty() : tensor<3xi64>
    %5 = linalg.fill ins(%c7_i64 : i64) outs(%4 : tensor<3xi64>) -> tensor<3xi64>
    %6:3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<9x3xi32>) outs(%1, %5, %3 : tensor<3xi32>, tensor<3xi64>, tensor<3xi64>) attrs =  {test.coupled_mixed} {
    ^bb0(%in: i32, %out: i32, %out_0: i64, %out_1: i64):
      %7 = linalg.index 0 : index
      %8 = arith.index_cast %7 : index to i64
      %9 = arith.maxsi %in, %out : i32
      %10 = arith.cmpi sgt, %in, %out : i32
      %11 = arith.cmpi eq, %in, %out : i32
      %12 = arith.cmpi slt, %8, %out_1 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.ori %10, %13 : i1
      %15 = arith.select %14, %8, %out_1 : i64
      %16 = arith.extsi %in : i32 to i64
      %17 = arith.addi %16, %out_0 : i64
      linalg.yield %9, %17, %15 : i32, i64, i64
    } -> (tensor<3xi32>, tensor<3xi64>, tensor<3xi64>)
    return %6#0, %6#1, %6#2 : tensor<3xi32>, tensor<3xi64>, tensor<3xi64>
  }
  func.func @empty_dynamic(%arg0: tensor<?x3xi32>) -> (tensor<3xi32>, tensor<3xi64>) {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %0 = tensor.empty() : tensor<3xi32>
    %1 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%0 : tensor<3xi32>) -> tensor<3xi32>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.fill ins(%c9223372036854775807_i64 : i64) outs(%2 : tensor<3xi64>) -> tensor<3xi64>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<?x3xi32>) outs(%1, %3 : tensor<3xi32>, tensor<3xi64>) attrs =  {test.coupled_pair} {
    ^bb0(%in: i32, %out: i32, %out_0: i64):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maxsi %in, %out : i32
      %8 = arith.cmpi sgt, %in, %out : i32
      %9 = arith.cmpi eq, %in, %out : i32
      %10 = arith.cmpi slt, %6, %out_0 : i64
      %11 = arith.andi %9, %10 : i1
      %12 = arith.ori %8, %11 : i1
      %13 = arith.select %12, %6, %out_0 : i64
      linalg.yield %7, %13 : i32, i64
    } -> (tensor<3xi32>, tensor<3xi64>)
    return %4#0, %4#1 : tensor<3xi32>, tensor<3xi64>
  }
  func.func private @check_empty() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<> : tensor<0x3xi32>
    %cast = tensor.cast %cst : tensor<0x3xi32> to tensor<?x3xi32>
    %0:2 = call @empty_dynamic(%cast) : (tensor<?x3xi32>) -> (tensor<3xi32>, tensor<3xi64>)
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %0#0[%c0] : tensor<3xi32>
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %1 = arith.cmpi ne, %extracted, %c-2147483648_i32 : i32
    %2 = arith.extui %1 : i1 to i32
    %3 = arith.addi %c0_i32, %2 : i32
    %c0_0 = arith.constant 0 : index
    %extracted_1 = tensor.extract %0#1[%c0_0] : tensor<3xi64>
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %4 = arith.cmpi ne, %extracted_1, %c9223372036854775807_i64 : i64
    %5 = arith.extui %4 : i1 to i32
    %6 = arith.addi %3, %5 : i32
    %c1 = arith.constant 1 : index
    %extracted_2 = tensor.extract %0#0[%c1] : tensor<3xi32>
    %c-2147483648_i32_3 = arith.constant -2147483648 : i32
    %7 = arith.cmpi ne, %extracted_2, %c-2147483648_i32_3 : i32
    %8 = arith.extui %7 : i1 to i32
    %9 = arith.addi %6, %8 : i32
    %c1_4 = arith.constant 1 : index
    %extracted_5 = tensor.extract %0#1[%c1_4] : tensor<3xi64>
    %c9223372036854775807_i64_6 = arith.constant 9223372036854775807 : i64
    %10 = arith.cmpi ne, %extracted_5, %c9223372036854775807_i64_6 : i64
    %11 = arith.extui %10 : i1 to i32
    %12 = arith.addi %9, %11 : i32
    %c2 = arith.constant 2 : index
    %extracted_7 = tensor.extract %0#0[%c2] : tensor<3xi32>
    %c-2147483648_i32_8 = arith.constant -2147483648 : i32
    %13 = arith.cmpi ne, %extracted_7, %c-2147483648_i32_8 : i32
    %14 = arith.extui %13 : i1 to i32
    %15 = arith.addi %12, %14 : i32
    %c2_9 = arith.constant 2 : index
    %extracted_10 = tensor.extract %0#1[%c2_9] : tensor<3xi64>
    %c9223372036854775807_i64_11 = arith.constant 9223372036854775807 : i64
    %16 = arith.cmpi ne, %extracted_10, %c9223372036854775807_i64_11 : i64
    %17 = arith.extui %16 : i1 to i32
    %18 = arith.addi %15, %17 : i32
    return %18 : i32
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<[[-9, -6, -3], [100, 100, 100], [5, 8, -8], [-7, -4, -1], [0, 3, 6], [7, -9, -6], [-5, -2, 1], [100, 100, 100], [9, -7, -4]]> : tensor<9x3xi32>
    %0:2 = call @signed_max(%cst) : (tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>)
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %0#0[%c0] : tensor<3xi32>
    %c100_i32 = arith.constant 100 : i32
    %1 = arith.cmpi ne, %extracted, %c100_i32 : i32
    %2 = arith.extui %1 : i1 to i32
    %3 = arith.addi %c0_i32, %2 : i32
    %c0_0 = arith.constant 0 : index
    %extracted_1 = tensor.extract %0#1[%c0_0] : tensor<3xi64>
    %c1_i64 = arith.constant 1 : i64
    %4 = arith.cmpi ne, %extracted_1, %c1_i64 : i64
    %5 = arith.extui %4 : i1 to i32
    %6 = arith.addi %3, %5 : i32
    %c1 = arith.constant 1 : index
    %extracted_2 = tensor.extract %0#0[%c1] : tensor<3xi32>
    %c100_i32_3 = arith.constant 100 : i32
    %7 = arith.cmpi ne, %extracted_2, %c100_i32_3 : i32
    %8 = arith.extui %7 : i1 to i32
    %9 = arith.addi %6, %8 : i32
    %c1_4 = arith.constant 1 : index
    %extracted_5 = tensor.extract %0#1[%c1_4] : tensor<3xi64>
    %c1_i64_6 = arith.constant 1 : i64
    %10 = arith.cmpi ne, %extracted_5, %c1_i64_6 : i64
    %11 = arith.extui %10 : i1 to i32
    %12 = arith.addi %9, %11 : i32
    %c2 = arith.constant 2 : index
    %extracted_7 = tensor.extract %0#0[%c2] : tensor<3xi32>
    %c100_i32_8 = arith.constant 100 : i32
    %13 = arith.cmpi ne, %extracted_7, %c100_i32_8 : i32
    %14 = arith.extui %13 : i1 to i32
    %15 = arith.addi %12, %14 : i32
    %c2_9 = arith.constant 2 : index
    %extracted_10 = tensor.extract %0#1[%c2_9] : tensor<3xi64>
    %c1_i64_11 = arith.constant 1 : i64
    %16 = arith.cmpi ne, %extracted_10, %c1_i64_11 : i64
    %17 = arith.extui %16 : i1 to i32
    %18 = arith.addi %15, %17 : i32
    %cst_12 = arith.constant dense<[[-9, -6, -3], [100, 100, 100], [5, 8, -8], [-7, -4, -1], [0, 3, 6], [7, -9, -6], [-5, -2, 1], [100, 100, 100], [9, -7, -4]]> : tensor<9x3xi32>
    %19:2 = call @winning_seed(%cst_12) : (tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>)
    %c0_13 = arith.constant 0 : index
    %extracted_14 = tensor.extract %19#0[%c0_13] : tensor<3xi32>
    %c200_i32 = arith.constant 200 : i32
    %20 = arith.cmpi ne, %extracted_14, %c200_i32 : i32
    %21 = arith.extui %20 : i1 to i32
    %22 = arith.addi %18, %21 : i32
    %c0_15 = arith.constant 0 : index
    %extracted_16 = tensor.extract %19#1[%c0_15] : tensor<3xi64>
    %c-7_i64 = arith.constant -7 : i64
    %23 = arith.cmpi ne, %extracted_16, %c-7_i64 : i64
    %24 = arith.extui %23 : i1 to i32
    %25 = arith.addi %22, %24 : i32
    %c1_17 = arith.constant 1 : index
    %extracted_18 = tensor.extract %19#0[%c1_17] : tensor<3xi32>
    %c200_i32_19 = arith.constant 200 : i32
    %26 = arith.cmpi ne, %extracted_18, %c200_i32_19 : i32
    %27 = arith.extui %26 : i1 to i32
    %28 = arith.addi %25, %27 : i32
    %c1_20 = arith.constant 1 : index
    %extracted_21 = tensor.extract %19#1[%c1_20] : tensor<3xi64>
    %c-7_i64_22 = arith.constant -7 : i64
    %29 = arith.cmpi ne, %extracted_21, %c-7_i64_22 : i64
    %30 = arith.extui %29 : i1 to i32
    %31 = arith.addi %28, %30 : i32
    %c2_23 = arith.constant 2 : index
    %extracted_24 = tensor.extract %19#0[%c2_23] : tensor<3xi32>
    %c200_i32_25 = arith.constant 200 : i32
    %32 = arith.cmpi ne, %extracted_24, %c200_i32_25 : i32
    %33 = arith.extui %32 : i1 to i32
    %34 = arith.addi %31, %33 : i32
    %c2_26 = arith.constant 2 : index
    %extracted_27 = tensor.extract %19#1[%c2_26] : tensor<3xi64>
    %c-7_i64_28 = arith.constant -7 : i64
    %35 = arith.cmpi ne, %extracted_27, %c-7_i64_28 : i64
    %36 = arith.extui %35 : i1 to i32
    %37 = arith.addi %34, %36 : i32
    %cst_29 = arith.constant dense<[[-9, -6, -3], [100, 100, 100], [5, 8, -8], [-7, -4, -1], [0, 3, 6], [7, -9, -6], [-5, -2, 1], [100, 100, 100], [9, -7, -4]]> : tensor<9x3xi32>
    %38:3 = call @mixed_sum(%cst_29) : (tensor<9x3xi32>) -> (tensor<3xi32>, tensor<3xi64>, tensor<3xi64>)
    %c0_30 = arith.constant 0 : index
    %extracted_31 = tensor.extract %38#0[%c0_30] : tensor<3xi32>
    %c100_i32_32 = arith.constant 100 : i32
    %39 = arith.cmpi ne, %extracted_31, %c100_i32_32 : i32
    %40 = arith.extui %39 : i1 to i32
    %41 = arith.addi %37, %40 : i32
    %c0_33 = arith.constant 0 : index
    %extracted_34 = tensor.extract %38#1[%c0_33] : tensor<3xi64>
    %c207_i64 = arith.constant 207 : i64
    %42 = arith.cmpi ne, %extracted_34, %c207_i64 : i64
    %43 = arith.extui %42 : i1 to i32
    %44 = arith.addi %41, %43 : i32
    %c0_35 = arith.constant 0 : index
    %extracted_36 = tensor.extract %38#2[%c0_35] : tensor<3xi64>
    %c1_i64_37 = arith.constant 1 : i64
    %45 = arith.cmpi ne, %extracted_36, %c1_i64_37 : i64
    %46 = arith.extui %45 : i1 to i32
    %47 = arith.addi %44, %46 : i32
    %c1_38 = arith.constant 1 : index
    %extracted_39 = tensor.extract %38#0[%c1_38] : tensor<3xi32>
    %c100_i32_40 = arith.constant 100 : i32
    %48 = arith.cmpi ne, %extracted_39, %c100_i32_40 : i32
    %49 = arith.extui %48 : i1 to i32
    %50 = arith.addi %47, %49 : i32
    %c1_41 = arith.constant 1 : index
    %extracted_42 = tensor.extract %38#1[%c1_41] : tensor<3xi64>
    %c190_i64 = arith.constant 190 : i64
    %51 = arith.cmpi ne, %extracted_42, %c190_i64 : i64
    %52 = arith.extui %51 : i1 to i32
    %53 = arith.addi %50, %52 : i32
    %c1_43 = arith.constant 1 : index
    %extracted_44 = tensor.extract %38#2[%c1_43] : tensor<3xi64>
    %c1_i64_45 = arith.constant 1 : i64
    %54 = arith.cmpi ne, %extracted_44, %c1_i64_45 : i64
    %55 = arith.extui %54 : i1 to i32
    %56 = arith.addi %53, %55 : i32
    %c2_46 = arith.constant 2 : index
    %extracted_47 = tensor.extract %38#0[%c2_46] : tensor<3xi32>
    %c100_i32_48 = arith.constant 100 : i32
    %57 = arith.cmpi ne, %extracted_47, %c100_i32_48 : i32
    %58 = arith.extui %57 : i1 to i32
    %59 = arith.addi %56, %58 : i32
    %c2_49 = arith.constant 2 : index
    %extracted_50 = tensor.extract %38#1[%c2_49] : tensor<3xi64>
    %c192_i64 = arith.constant 192 : i64
    %60 = arith.cmpi ne, %extracted_50, %c192_i64 : i64
    %61 = arith.extui %60 : i1 to i32
    %62 = arith.addi %59, %61 : i32
    %c2_51 = arith.constant 2 : index
    %extracted_52 = tensor.extract %38#2[%c2_51] : tensor<3xi64>
    %c1_i64_53 = arith.constant 1 : i64
    %63 = arith.cmpi ne, %extracted_52, %c1_i64_53 : i64
    %64 = arith.extui %63 : i1 to i32
    %65 = arith.addi %62, %64 : i32
    %66 = call @check_empty() : () -> i32
    %67 = arith.addi %65, %66 : i32
    return %67 : i32
  }
}
