// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @truncf_f16_to_bf8
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf16>
llvm.func @truncf_f16_to_bf8(%src: vector<16xf16>) -> vector<16xi8> {
  // CHECK:  %[[VAR0:.*]] = llvm.shufflevector %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xf16> 
  // CHECK:  %[[VAR1:.*]] = llvm.shufflevector %[[ARG0]], %[[ARG0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf16> 
  // CHECK:  %[[VAR2:.*]] = llvm.bitcast %[[VAR0]] : vector<8xf16> to vector<16xi8>
  // CHECK:  %[[VAR3:.*]] = llvm.bitcast %[[VAR1]] : vector<8xf16> to vector<16xi8>
  // CHECK:  %[[VAR4:.*]] = llvm.shufflevector %[[VAR2]], %[[VAR2]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi8> 
  // CHECK:  %[[VAR5:.*]] = llvm.shufflevector %[[VAR3]], %[[VAR3]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi8> 
  // CHECK:  %[[VAR6:.*]] = llvm.shufflevector %4, %5 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8> 
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = bf8 } : (vector<16xf16>) -> vector<16xi8>
  llvm.return %dst : vector<16xi8>
}

// -----

// CHECK-LABEL: llvm.func spir_funccc @_Z49intel_sub_group_bf8_bf8_scaled_matrix_mad_k32_f32Dv8_sDv8_iDv8_fcc
// CHECK-SAME: (vector<8xi16>, vector<8xi32>, vector<8xf32>, i8, i8) -> vector<8xf32>
// CHECK-SAME:   attributes {convergent, memory_effects = #llvm.memory_effects<other = none,
// CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
// CHECK-SAME:   targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
// CHECK: llvm.func @mma_mx_bf8_bf8_k32_f32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi16>, %[[ARG1:.*]]: vector<8xi32>
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: vector<8xf32>
llvm.func @mma_mx_bf8_bf8_k32_f32(%a: vector<8xi16>, %b: vector<8xi32>, %scale_a: i8, %scale_b: i8, %c: vector<8xf32>) -> vector<8xf32> {
  // CHECK:  %[[VAR0:.*]] = llvm.call spir_funccc @_Z49intel_sub_group_bf8_bf8_scaled_matrix_mad_k32_f32Dv8_sDv8_iDv8_fcc
  // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[ARG4]], %[[ARG2]], %[[ARG3]])
  // CHECK-SAME: {convergent, function_type = !llvm.func<vector<8xf32> (vector<8xi16>, vector<8xi32>, vector<8xf32>, i8, i8)>,
  // CHECK-SAME: linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none,
  // CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
  // CHECK-SAME:   targetMem0 = none, targetMem1 = none>,
  // CHECK-SAME: no_unwind, sym_name = "_Z49intel_sub_group_bf8_bf8_scaled_matrix_mad_k32_f32Dv8_sDv8_iDv8_fcc",
  // CHECK-SAME: visibility_ = 0 : i64, will_return} :
  // CHECK-SAME: (vector<8xi16>, vector<8xi32>, vector<8xf32>, i8, i8) -> vector<8xf32>
  %result = xevm.mma_mx %a, %b, %scale_a, %scale_b, %c
          {shape=<m=8, n=16, k=32>, types=<d=f32, a=bf8, b=bf8, c=f32>}
          : (vector<8xi16>, vector<8xi32>, i8, i8, vector<8xf32>) -> vector<8xf32>
  llvm.return %result : vector<8xf32>
}
