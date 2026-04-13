// RUN: mlir-opt --convert-amdgpu-to-rocdl=chipset=gfx950 --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @test_permlane16_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @test_permlane16_i32(%arg0 : i32) -> i32 {
// CHECK:  %[[PERM:.*]] = rocdl.permlane16.swap %[[ARG0]], %[[ARG0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[ARG0]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  return %[[SEL]] : i32
  %0 = amdgpu.permlane_swap %arg0 16 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_permlane16_i32_optional_attr
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @test_permlane16_i32_optional_attr(%arg0 : i32) -> i32 {
// CHECK:  %[[PERM:.*]] = rocdl.permlane16.swap %[[ARG0]], %[[ARG0]], true, true : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[ARG0]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  return %[[SEL]] : i32
  %0 = amdgpu.permlane_swap %arg0 16 { fetch_inactive = true, bound_ctrl = true }  : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_permlane32_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @test_permlane32_i32(%arg0 : i32) -> i32 {
// CHECK:  %[[PERM:.*]] = rocdl.permlane32.swap %[[ARG0]], %[[ARG0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[ARG0]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  return %[[SEL]] : i32
  %0 = amdgpu.permlane_swap %arg0 32 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_permlane16_f32
// CHECK-SAME: (%[[ARG0:.*]]: f32)
func.func @test_permlane16_f32(%arg0 : f32) -> f32 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f32 to i32
// CHECK:  %[[PERM:.*]] = rocdl.permlane16.swap %[[CAST]], %[[CAST]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[CAST]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[SEL]] : i32 to f32
// CHECK:  return %[[RES_CAST]] : f32
  %0 = amdgpu.permlane_swap %arg0 16 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_permlane32_f32
// CHECK-SAME: (%[[ARG0:.*]]: f32)
func.func @test_permlane32_f32(%arg0 : f32) -> f32 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f32 to i32
// CHECK:  %[[PERM:.*]] = rocdl.permlane32.swap %[[CAST]], %[[CAST]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[CAST]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[SEL]] : i32 to f32
// CHECK:  return %[[RES_CAST]] : f32
  %0 = amdgpu.permlane_swap %arg0 32 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_permlane16_f16
// CHECK-SAME: (%[[ARG0:.*]]: f16)
func.func @test_permlane16_f16(%arg0 : f16) -> f16 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f16 to i16
// CHECK:  %[[ZEXT:.*]] = llvm.zext %[[CAST]] : i16 to i32
// CHECK:  %[[PERM:.*]] = rocdl.permlane16.swap %[[ZEXT]], %[[ZEXT]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[ZEXT]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  %[[TRUNC:.*]] = llvm.trunc %[[SEL]] : i32 to i16
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to f16
// CHECK:  return %[[RES_CAST]] : f16
  %0 = amdgpu.permlane_swap %arg0 16 : f16
  return %0 : f16
}

// CHECK-LABEL: func @test_permlane32_f16
// CHECK-SAME: (%[[ARG0:.*]]: f16)
func.func @test_permlane32_f16(%arg0 : f16) -> f16 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : f16 to i16
// CHECK:  %[[ZEXT:.*]] = llvm.zext %[[CAST]] : i16 to i32
// CHECK:  %[[PERM:.*]] = rocdl.permlane32.swap %[[ZEXT]], %[[ZEXT]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:  %[[E0:.*]] = llvm.extractvalue %[[PERM]][0] : !llvm.struct<(i32, i32)>
// CHECK:  %[[E1:.*]] = llvm.extractvalue %[[PERM]][1] : !llvm.struct<(i32, i32)>
// CHECK:  %[[CMP:.*]] = llvm.icmp "eq" %[[E0]], %[[ZEXT]] : i32
// CHECK:  %[[SEL:.*]] = llvm.select %[[CMP]], %[[E1]], %[[E0]] : i1, i32
// CHECK:  %[[TRUNC:.*]] = llvm.trunc %[[SEL]] : i32 to i16
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to f16
// CHECK:  return %[[RES_CAST]] : f16
  %0 = amdgpu.permlane_swap %arg0 32 : f16
  return %0 : f16
}

// CHECK-LABEL: func @test_permlane16_2xi32
// CHECK-SAME: (%[[ARG0:.*]]: vector<2xi32>)
func.func @test_permlane16_2xi32(%arg0 : vector<2xi32>) -> vector<2xi32> {
// CHECK-DAG:  %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[ELEM0:.*]] = llvm.extractelement %[[ARG0]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[ELEM1:.*]] = llvm.extractelement %[[ARG0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[T0:.*]] = rocdl.permlane16.swap %[[ELEM0]], %[[ELEM0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[T0_0:.*]] = llvm.extractvalue %[[T0]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[T0_1:.*]] = llvm.extractvalue %[[T0]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP0:.*]] = llvm.icmp "eq" %[[T0_0]], %[[ELEM0]] : i32
// CHECK:      %[[PERM0:.*]] = llvm.select %[[CMP0]], %[[T0_1]], %[[T0_0]] : i1, i32
// CHECK:      %[[T1:.*]] = rocdl.permlane16.swap %[[ELEM1]], %[[ELEM1]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[T1_0:.*]] = llvm.extractvalue %[[T1]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[T1_1:.*]] = llvm.extractvalue %[[T1]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP1:.*]] = llvm.icmp "eq" %[[T1_0]], %[[ELEM1]] : i32
// CHECK:      %[[PERM1:.*]] = llvm.select %[[CMP1]], %[[T1_1]], %[[T1_0]] : i1, i32
// CHECK:      %[[VEC_INSERT0:.*]] = llvm.insertelement %[[PERM0]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[VEC_INSERT1:.*]] = llvm.insertelement %[[PERM1]], %[[VEC_INSERT0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      return %[[VEC_INSERT1]] : vector<2xi32>
  %0 = amdgpu.permlane_swap %arg0 16 : vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL: func @test_permlane32_2xi32
// CHECK-SAME: (%[[ARG0:.*]]: vector<2xi32>)
func.func @test_permlane32_2xi32(%arg0 : vector<2xi32>) -> vector<2xi32> {
// CHECK-DAG:  %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[ELEM0:.*]] = llvm.extractelement %[[ARG0]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[ELEM1:.*]] = llvm.extractelement %[[ARG0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[T0:.*]] = rocdl.permlane32.swap %[[ELEM0]], %[[ELEM0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[T0_0:.*]] = llvm.extractvalue %[[T0]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[T0_1:.*]] = llvm.extractvalue %[[T0]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP0:.*]] = llvm.icmp "eq" %[[T0_0]], %[[ELEM0]] : i32
// CHECK:      %[[PERM0:.*]] = llvm.select %[[CMP0]], %[[T0_1]], %[[T0_0]] : i1, i32
// CHECK:      %[[T1:.*]] = rocdl.permlane32.swap %[[ELEM1]], %[[ELEM1]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[T1_0:.*]] = llvm.extractvalue %[[T1]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[T1_1:.*]] = llvm.extractvalue %[[T1]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP1:.*]] = llvm.icmp "eq" %[[T1_0]], %[[ELEM1]] : i32
// CHECK:      %[[PERM1:.*]] = llvm.select %[[CMP1]], %[[T1_1]], %[[T1_0]] : i1, i32
// CHECK:      %[[VEC_INSERT0:.*]] = llvm.insertelement %[[PERM0]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[VEC_INSERT1:.*]] = llvm.insertelement %[[PERM1]], %[[VEC_INSERT0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      return %[[VEC_INSERT1]] : vector<2xi32>
  %0 = amdgpu.permlane_swap %arg0 32 : vector<2xi32>
  return %0 : vector<2xi32>
}

// CHECK-LABEL: func @test_permlane16_4xf16
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf16>)
func.func @test_permlane16_4xf16(%arg0 : vector<4xf16>) -> vector<4xf16> {
// CHECK-DAG:  %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[CAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
// CHECK:      %[[ELEM0:.*]] = llvm.extractelement %[[CAST1]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[ELEM1:.*]] = llvm.extractelement %[[CAST1]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[PERM0_TUPLE:.*]] = rocdl.permlane16.swap %[[ELEM0]], %[[ELEM0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[PERM0_E0:.*]] = llvm.extractvalue %[[PERM0_TUPLE]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[PERM0_E1:.*]] = llvm.extractvalue %[[PERM0_TUPLE]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP0:.*]] = llvm.icmp "eq" %[[PERM0_E0]], %[[ELEM0]] : i32
// CHECK:      %[[PERM0:.*]] = llvm.select %[[CMP0]], %[[PERM0_E1]], %[[PERM0_E0]] : i1, i32
// CHECK:      %[[PERM1_TUPLE:.*]] = rocdl.permlane16.swap %[[ELEM1]], %[[ELEM1]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[PERM1_E0:.*]] = llvm.extractvalue %[[PERM1_TUPLE]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[PERM1_E1:.*]] = llvm.extractvalue %[[PERM1_TUPLE]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP1:.*]] = llvm.icmp "eq" %[[PERM1_E0]], %[[ELEM1]] : i32
// CHECK:      %[[PERM1:.*]] = llvm.select %[[CMP1]], %[[PERM1_E1]], %[[PERM1_E0]] : i1, i32
// CHECK:      %[[VEC_INSERT0:.*]] = llvm.insertelement %[[PERM0]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[VEC_INSERT1:.*]] = llvm.insertelement %[[PERM1]], %[[VEC_INSERT0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[CAST2:.*]] = llvm.bitcast %[[VEC_INSERT1]] : vector<2xi32> to vector<4xf16>
// CHECK:      return %[[CAST2]] : vector<4xf16>
  %0 = amdgpu.permlane_swap %arg0 16 : vector<4xf16>
  return %0 : vector<4xf16>
}

// CHECK-LABEL: func @test_permlane32_4xf16
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf16>)
func.func @test_permlane32_4xf16(%arg0 : vector<4xf16>) -> vector<4xf16> {
// CHECK-DAG:  %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[CAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<4xf16> to vector<2xi32>
// CHECK:      %[[ELEM0:.*]] = llvm.extractelement %[[CAST1]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[ELEM1:.*]] = llvm.extractelement %[[CAST1]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[PERM0_TUPLE:.*]] = rocdl.permlane32.swap %[[ELEM0]], %[[ELEM0]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[PERM0_E0:.*]] = llvm.extractvalue %[[PERM0_TUPLE]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[PERM0_E1:.*]] = llvm.extractvalue %[[PERM0_TUPLE]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP0:.*]] = llvm.icmp "eq" %[[PERM0_E0]], %[[ELEM0]] : i32
// CHECK:      %[[PERM0:.*]] = llvm.select %[[CMP0]], %[[PERM0_E1]], %[[PERM0_E0]] : i1, i32
// CHECK:      %[[PERM1_TUPLE:.*]] = rocdl.permlane32.swap %[[ELEM1]], %[[ELEM1]], false, false : (i32, i32) -> <(i32, i32)>
// CHECK:      %[[PERM1_E0:.*]] = llvm.extractvalue %[[PERM1_TUPLE]][0] : !llvm.struct<(i32, i32)>
// CHECK:      %[[PERM1_E1:.*]] = llvm.extractvalue %[[PERM1_TUPLE]][1] : !llvm.struct<(i32, i32)>
// CHECK:      %[[CMP1:.*]] = llvm.icmp "eq" %[[PERM1_E0]], %[[ELEM1]] : i32
// CHECK:      %[[PERM1:.*]] = llvm.select %[[CMP1]], %[[PERM1_E1]], %[[PERM1_E0]] : i1, i32
// CHECK:      %[[VEC_INSERT0:.*]] = llvm.insertelement %[[PERM0]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[VEC_INSERT1:.*]] = llvm.insertelement %[[PERM1]], %[[VEC_INSERT0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[CAST2:.*]] = llvm.bitcast %[[VEC_INSERT1]] : vector<2xi32> to vector<4xf16>
// CHECK:      return %[[CAST2]] : vector<4xf16>
  %0 = amdgpu.permlane_swap %arg0 32 : vector<4xf16>
  return %0 : vector<4xf16>
}
