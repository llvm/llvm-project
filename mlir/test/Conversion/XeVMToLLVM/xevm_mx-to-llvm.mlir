// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @__builtin_IB_hftobf8_16(vector<16xf16>) -> vector<16xi8>
// CHECK-SAME: attributes {convergent, memory_effects = #llvm.memory_effects<other = none,
// CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
// CHECK-SAME:   targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
// CHECK-LABEL: llvm.func @truncf_f16_to_bf8
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf16>
llvm.func @truncf_f16_to_bf8(%src: vector<16xf16>) -> vector<16xi8> {
  // CHECK: %[[VAR0:.*]] = llvm.call spir_funccc @__builtin_IB_hftobf8_16(%[[ARG0]])
  // CHECK-SAME: {convergent, function_type = !llvm.func<vector<16xi8> (vector<16xf16>)>,
  // CHECK-SAME: linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none,
  // CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
  // CHECK-SAME:   targetMem0 = none, targetMem1 = none>,
  // CHECK-SAME: no_unwind, sym_name = "__builtin_IB_hftobf8_16",
  // CHECK-SAME: visibility_ = 0 : i64, will_return} :
  // CHECK-SAME: (vector<16xf16>) -> vector<16xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = bf8 } : (vector<16xf16>) -> vector<16xi8>
  llvm.return %dst : vector<16xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_hftohf8_16(vector<16xf16>) -> vector<16xi8>
// CHECK-SAME: attributes {convergent, memory_effects = #llvm.memory_effects<other = none,
// CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
// CHECK-SAME:   targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
// CHECK-LABEL: llvm.func @truncf_f16_to_hf8
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf16>
llvm.func @truncf_f16_to_hf8(%src: vector<16xf16>) -> vector<16xi8> {
  // CHECK: %[[VAR0:.*]] = llvm.call spir_funccc @__builtin_IB_hftohf8_16(%[[ARG0]])
  // CHECK-SAME: {convergent, function_type = !llvm.func<vector<16xi8> (vector<16xf16>)>,
  // CHECK-SAME: linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none,
  // CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
  // CHECK-SAME:   targetMem0 = none, targetMem1 = none>,
  // CHECK-SAME: no_unwind, sym_name = "__builtin_IB_hftohf8_16",
  // CHECK-SAME: visibility_ = 0 : i64, will_return} :
  // CHECK-SAME: (vector<16xf16>) -> vector<16xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = f8 } : (vector<16xf16>) -> vector<16xi8>
  llvm.return %dst : vector<16xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_hftobf8_16(vector<16xf16>) -> vector<16xi8>
// CHECK: llvm.func spir_funccc @_Z14convert_half16Dv16_f(vector<16xf32>) -> vector<16xf16>
// CHECK: llvm.func spir_funccc @__builtin_IB_bftof_16(vector<16xi16>) -> vector<16xf32>
// CHECK-LABEL: llvm.func @truncf_bf16_to_bf8
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
llvm.func @truncf_bf16_to_bf8(%src: vector<16xbf16>) -> vector<16xi8> {
  // CHECK: %[[VAR0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<16xi16>
  // CHECK: %[[VAR1:.*]] = llvm.call spir_funccc @__builtin_IB_bftof_16(%[[VAR0]])
  // CHECK-SAME: : (vector<16xi16>) -> vector<16xf32>
  // CHECK: %[[VAR2:.*]] = llvm.call spir_funccc @_Z14convert_half16Dv16_f(%[[VAR1]])
  // CHECK-SAME: : (vector<16xf32>) -> vector<16xf16>
  // CHECK: %[[VAR3:.*]] = llvm.call spir_funccc @__builtin_IB_hftobf8_16(%[[VAR2]])
  // CHECK-SAME: : (vector<16xf16>) -> vector<16xi8>
  %dst = xevm.truncf %src { src_etype = bf16, dst_etype = bf8 } : (vector<16xbf16>) -> vector<16xi8>
  llvm.return %dst : vector<16xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_hftohf8_16(vector<16xf16>) -> vector<16xi8>
// CHECK: llvm.func spir_funccc @_Z14convert_half16Dv16_f(vector<16xf32>) -> vector<16xf16>
// CHECK: llvm.func spir_funccc @__builtin_IB_bftof_16(vector<16xi16>) -> vector<16xf32>
// CHECK-LABEL: llvm.func @truncf_bf16_to_hf8
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
llvm.func @truncf_bf16_to_hf8(%src: vector<16xbf16>) -> vector<16xi8> {
  // CHECK: %[[VAR0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<16xi16>
  // CHECK: %[[VAR1:.*]] = llvm.call spir_funccc @__builtin_IB_bftof_16(%[[VAR0]])
  // CHECK-SAME: : (vector<16xi16>) -> vector<16xf32>
  // CHECK: %[[VAR2:.*]] = llvm.call spir_funccc @_Z14convert_half16Dv16_f(%[[VAR1]])
  // CHECK-SAME: : (vector<16xf32>) -> vector<16xf16>
  // CHECK: %[[VAR3:.*]] = llvm.call spir_funccc @__builtin_IB_hftohf8_16(%[[VAR2]])
  // CHECK-SAME: : (vector<16xf16>) -> vector<16xi8>
  %dst = xevm.truncf %src { src_etype = bf16, dst_etype = f8 } : (vector<16xbf16>) -> vector<16xi8>
  llvm.return %dst : vector<16xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_hf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_f16_to_e2m1
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf16>
llvm.func @truncf_f16_to_e2m1(%src: vector<16xf16>) -> vector<8xi8> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: %[[C6:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK: %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BC:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf16> to vector<8xi32>
  // CHECK: %[[E0:.*]] = llvm.extractelement %[[BC]][%[[C0]] : i32] : vector<8xi32>
  // CHECK: %[[E2:.*]] = llvm.extractelement %[[BC]][%[[C2]] : i32] : vector<8xi32>
  // CHECK: %[[CALL0:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[E0]], %[[E2]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[E1:.*]] = llvm.extractelement %[[BC]][%[[C1]] : i32] : vector<8xi32>
  // CHECK: %[[E3:.*]] = llvm.extractelement %[[BC]][%[[C3]] : i32] : vector<8xi32>
  // CHECK: %[[CALL1:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[E1]], %[[E3]], %[[C1]], %[[C2]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[OR0:.*]] = llvm.or %[[CALL0]], %[[CALL1]] : i32
  // CHECK: %[[E4:.*]] = llvm.extractelement %[[BC]][%[[C4]] : i32] : vector<8xi32>
  // CHECK: %[[E6:.*]] = llvm.extractelement %[[BC]][%[[C6]] : i32] : vector<8xi32>
  // CHECK: %[[CALL2:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[E4]], %[[E6]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[E5:.*]] = llvm.extractelement %[[BC]][%[[C5]] : i32] : vector<8xi32>
  // CHECK: %[[E7:.*]] = llvm.extractelement %[[BC]][%[[C7]] : i32] : vector<8xi32>
  // CHECK: %[[CALL3:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_hf16(%[[E5]], %[[E7]], %[[C1]], %[[C2]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[OR1:.*]] = llvm.or %[[CALL2]], %[[CALL3]] : i32
  // CHECK: %[[INS0:.*]] = llvm.insertelement %[[OR0]], %[[UNDEF]][%[[C0]] : i32] : vector<2xi32>
  // CHECK: %[[INS1:.*]] = llvm.insertelement %[[OR1]], %[[INS0]][%[[C1]] : i32] : vector<2xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[INS1]] : vector<2xi32> to vector<8xi8>
  %dst = xevm.truncf %src { src_etype = f16, dst_etype = e2m1 } : (vector<16xf16>) -> vector<8xi8>
  llvm.return %dst : vector<8xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_dnscl_bf16(i32, i32, i32, i32) -> i32
// CHECK-LABEL: llvm.func @truncf_bf16_to_e2m1
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
llvm.func @truncf_bf16_to_e2m1(%src: vector<16xbf16>) -> vector<8xi8> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: %[[C6:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK: %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BC:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<8xi32>
  // CHECK: %[[E0:.*]] = llvm.extractelement %[[BC]][%[[C0]] : i32] : vector<8xi32>
  // CHECK: %[[E2:.*]] = llvm.extractelement %[[BC]][%[[C2]] : i32] : vector<8xi32>
  // CHECK: %[[CALL0:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[E0]], %[[E2]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[E1:.*]] = llvm.extractelement %[[BC]][%[[C1]] : i32] : vector<8xi32>
  // CHECK: %[[E3:.*]] = llvm.extractelement %[[BC]][%[[C3]] : i32] : vector<8xi32>
  // CHECK: %[[CALL1:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[E1]], %[[E3]], %[[C1]], %[[C2]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[OR0:.*]] = llvm.or %[[CALL0]], %[[CALL1]] : i32
  // CHECK: %[[E4:.*]] = llvm.extractelement %[[BC]][%[[C4]] : i32] : vector<8xi32>
  // CHECK: %[[E6:.*]] = llvm.extractelement %[[BC]][%[[C6]] : i32] : vector<8xi32>
  // CHECK: %[[CALL2:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[E4]], %[[E6]], %[[C1]], %[[C0]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[E5:.*]] = llvm.extractelement %[[BC]][%[[C5]] : i32] : vector<8xi32>
  // CHECK: %[[E7:.*]] = llvm.extractelement %[[BC]][%[[C7]] : i32] : vector<8xi32>
  // CHECK: %[[CALL3:.*]] = llvm.call spir_funccc @__builtin_IB_dnscl_bf16(%[[E5]], %[[E7]], %[[C1]], %[[C2]])
  // CHECK-SAME: : (i32, i32, i32, i32) -> i32
  // CHECK: %[[OR1:.*]] = llvm.or %[[CALL2]], %[[CALL3]] : i32
  // CHECK: %[[INS0:.*]] = llvm.insertelement %[[OR0]], %[[UNDEF]][%[[C0]] : i32] : vector<2xi32>
  // CHECK: %[[INS1:.*]] = llvm.insertelement %[[OR1]], %[[INS0]][%[[C1]] : i32] : vector<2xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[INS1]] : vector<2xi32> to vector<8xi8>
  %dst = xevm.truncf %src { src_etype = bf16, dst_etype = e2m1 } : (vector<16xbf16>) -> vector<8xi8>
  llvm.return %dst : vector<8xi8>
}

// -----

// CHECK: llvm.func spir_funccc @__builtin_IB_sub_group16_bdpas_f_f_bf8_bf8_8_8
// CHECK-SAME: (vector<8xf32>, vector<8xi16>, vector<8xi32>, i8, i8) -> vector<8xf32>
// CHECK-SAME:   attributes {convergent, memory_effects = #llvm.memory_effects<other = none,
// CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
// CHECK-SAME:   targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
// CHECK: llvm.func @mma_mx_bf8_bf8_k32_f32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi16>, %[[ARG1:.*]]: vector<8xi32>
// CHECK-SAME: %[[ARG2:.*]]: i8, %[[ARG3:.*]]: i8, %[[ARG4:.*]]: vector<8xf32>
llvm.func @mma_mx_bf8_bf8_k32_f32(%a: vector<8xi16>, %b: vector<8xi32>, %scale_a: i8, %scale_b: i8, %c: vector<8xf32>) -> vector<8xf32> {
  // CHECK: %[[VAR0:.*]] = llvm.call spir_funccc @__builtin_IB_sub_group16_bdpas_f_f_bf8_bf8_8_8
  // CHECK-SAME: (%[[ARG4]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
  // CHECK-SAME: {convergent, function_type = !llvm.func<vector<8xf32> (vector<8xf32>, vector<8xi16>, vector<8xi32>, i8, i8)>,
  // CHECK-SAME: linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none,
  // CHECK-SAME:   argMem = none, inaccessibleMem = none, errnoMem = none,
  // CHECK-SAME:   targetMem0 = none, targetMem1 = none>,
  // CHECK-SAME: no_unwind, sym_name = "__builtin_IB_sub_group16_bdpas_f_f_bf8_bf8_8_8",
  // CHECK-SAME: visibility_ = 0 : i64, will_return} :
  // CHECK-SAME: (vector<8xf32>, vector<8xi16>, vector<8xi32>, i8, i8) -> vector<8xf32>
  %result = xevm.mma_mx %a, %b, %scale_a, %scale_b, %c
          {shape=<m=8, n=16, k=32>, types=<d=f32, a=bf8, b=bf8, c=f32>}
          : (vector<8xi16>, vector<8xi32>, i8, i8, vector<8xf32>) -> vector<8xf32>
  llvm.return %result : vector<8xf32>
}
