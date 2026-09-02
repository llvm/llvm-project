// RUN: mlir-opt --convert-amdgpu-to-rocdl=chipset=gfx1200 --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @test_permlane_var_i32
// CHECK-SAME: (%[[SRC:.*]]: i32, %[[SEL:.*]]: i32)
func.func @test_permlane_var_i32(%src : i32, %sel : i32) -> i32 {
// CHECK:  %[[RES:.*]] = rocdl.permlane16.var %[[SRC]], %[[SRC]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:  return %[[RES]] : i32
  %0 = amdgpu.permlane_var %src, %sel : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_permlane_var_cross_i32
// CHECK-SAME: (%[[SRC:.*]]: i32, %[[SEL:.*]]: i32)
func.func @test_permlane_var_cross_i32(%src : i32, %sel : i32) -> i32 {
// CHECK:  %[[RES:.*]] = rocdl.permlanex16.var %[[SRC]], %[[SRC]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:  return %[[RES]] : i32
  %0 = amdgpu.permlane_var %src, %sel { cross = true } : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_permlane_var_f32
// CHECK-SAME: (%[[SRC:.*]]: f32, %[[SEL:.*]]: i32)
func.func @test_permlane_var_f32(%src : f32, %sel : i32) -> f32 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[SRC]] : f32 to i32
// CHECK:  %[[RES:.*]] = rocdl.permlane16.var %[[CAST]], %[[CAST]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[RES]] : i32 to f32
// CHECK:  return %[[RES_CAST]] : f32
  %0 = amdgpu.permlane_var %src, %sel : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_permlane_var_f16
// CHECK-SAME: (%[[SRC:.*]]: f16, %[[SEL:.*]]: i32)
func.func @test_permlane_var_f16(%src : f16, %sel : i32) -> f16 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[SRC]] : f16 to i16
// CHECK:  %[[ZEXT:.*]] = llvm.zext %[[CAST]] : i16 to i32
// CHECK:  %[[RES:.*]] = rocdl.permlane16.var %[[ZEXT]], %[[ZEXT]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:  %[[TRUNC:.*]] = llvm.trunc %[[RES]] : i32 to i16
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to f16
// CHECK:  return %[[RES_CAST]] : f16
  %0 = amdgpu.permlane_var %src, %sel : f16
  return %0 : f16
}

// CHECK-LABEL: func @test_permlane_var_cross_f16
// CHECK-SAME: (%[[SRC:.*]]: f16, %[[SEL:.*]]: i32)
func.func @test_permlane_var_cross_f16(%src : f16, %sel : i32) -> f16 {
// CHECK:  %[[CAST:.*]] = llvm.bitcast %[[SRC]] : f16 to i16
// CHECK:  %[[ZEXT:.*]] = llvm.zext %[[CAST]] : i16 to i32
// CHECK:  %[[RES:.*]] = rocdl.permlanex16.var %[[ZEXT]], %[[ZEXT]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:  %[[TRUNC:.*]] = llvm.trunc %[[RES]] : i32 to i16
// CHECK:  %[[RES_CAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to f16
// CHECK:  return %[[RES_CAST]] : f16
  %0 = amdgpu.permlane_var %src, %sel { cross = true } : f16
  return %0 : f16
}

// CHECK-LABEL: func @test_permlane_var_4xf16
// CHECK-SAME: (%[[SRC:.*]]: vector<4xf16>, %[[SEL:.*]]: i32)
func.func @test_permlane_var_4xf16(%src : vector<4xf16>, %sel : i32) -> vector<4xf16> {
// CHECK-DAG:  %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[CAST:.*]] = llvm.bitcast %[[SRC]] : vector<4xf16> to vector<2xi32>
// CHECK:      %[[ELEM0:.*]] = llvm.extractelement %[[CAST]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[ELEM1:.*]] = llvm.extractelement %[[CAST]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[PERM0:.*]] = rocdl.permlane16.var %[[ELEM0]], %[[ELEM0]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:      %[[PERM1:.*]] = rocdl.permlane16.var %[[ELEM1]], %[[ELEM1]], %[[SEL]], false, false : (i32, i32, i32) -> i32
// CHECK:      %[[INS0:.*]] = llvm.insertelement %[[PERM0]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:      %[[INS1:.*]] = llvm.insertelement %[[PERM1]], %[[INS0]][%[[C1]] : i32] : vector<2xi32>
// CHECK:      %[[RES:.*]] = llvm.bitcast %[[INS1]] : vector<2xi32> to vector<4xf16>
// CHECK:      return %[[RES]] : vector<4xf16>
  %0 = amdgpu.permlane_var %src, %sel : vector<4xf16>
  return %0 : vector<4xf16>
}

// CHECK-LABEL: func @test_permlane_var_attrs
// CHECK-SAME: (%[[SRC:.*]]: i32, %[[SEL:.*]]: i32)
func.func @test_permlane_var_attrs(%src : i32, %sel : i32) -> i32 {
// CHECK:  %[[RES:.*]] = rocdl.permlanex16.var %[[SRC]], %[[SRC]], %[[SEL]], true, true : (i32, i32, i32) -> i32
// CHECK:  return %[[RES]] : i32
  %0 = amdgpu.permlane_var %src, %sel { cross = true, fetch_inactive = true, bound_ctrl = true } : i32
  return %0 : i32
}
