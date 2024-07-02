// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx908 %s | FileCheck %s
// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx90a %s | FileCheck %s
// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx942 %s | FileCheck %s

func.func @test_dpp(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: func @test_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 257, 10, 15, false : i32
  // CHECK: return %0 : i32
  %0 = amdgpu.dpp %arg0 %arg1 row_shl ( 0x1 : i32 ) { row_mask = 0xa : i32, bound_ctrl = false } : i32
    return %0 : i32
}

func.func @quad_dpp(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: func @quad_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 145, 1, 1, true : i32
  // CHECK: return %0 : i32
  %0 = amdgpu.dpp %arg0 %arg1 quad_perm ( [1,0,1,2] ) { row_mask = 0x1 : i32, bank_mask = 0x1 : i32, bound_ctrl = true } : i32
    return %0 : i32
}

func.func @wave_shr_dpp(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: func @wave_shr_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 312, 10, 1, true : i32
  // CHECK: return %0 : i32
  %0 = amdgpu.dpp %arg0 %arg1 wave_shr { row_mask = 0xa : i32, bank_mask = 0x1 : i32, bound_ctrl = true } : i32
    return %0 : i32
}

func.func @row_half_mirror_update_dpp(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: func @row_half_mirror_update_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 321, 15, 1, false : i32
  // CHECK: return %0 : i32
%0 = amdgpu.dpp %arg0 %arg1 row_half_mirror { bank_mask = 0x1 : i32 } : i32
    return %0 : i32
}

func.func @wave_rol_update_dpp(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: func @wave_rol_update_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 308, 10, 1, false : i32
  // CHECK: return %0 : i32
  %0 = amdgpu.dpp %arg0 %arg1 wave_rol { row_mask = 0xa : i32, bank_mask = 0x1 : i32 } : i32
    return %0 : i32
}

func.func @row_bcast_dpp_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK-LABEL: func @row_bcast_dpp_f32
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 322, 15, 15, true : f32
  // CHECK: return %0 : f32
  %0 = amdgpu.dpp %arg0 %arg1 row_bcast_15 { bound_ctrl = true } : f32
    return %0 : f32
}

func.func @test_dpp_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK-LABEL: func @test_dpp_f32
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 320, 1, 4, true : f32
  // CHECK: return %0 : f32
  %0 = amdgpu.dpp %arg0 %arg1 row_mirror { row_mask = 0x1 : i32, bank_mask = 0x4 : i32, bound_ctrl = true } : f32
    return %0 : f32
}

func.func @quad_perm_update_dpp_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK-LABEL: func @quad_perm_update_dpp_f32
  // CHECK: rocdl.update.dpp %arg0, %arg1 with  88, 15, 1, false : f32
  // CHECK: return %0 : f32
  %0 = amdgpu.dpp %arg0 %arg1 quad_perm ( [0,2,1,1] ) { bank_mask = 0x1 : i32 } : f32
    return %0 : f32
}

func.func @quad_perm_dpp(%arg0: i64, %arg1: i64) -> i64 {
  // CHECK-LABEL: func @quad_perm_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 88, 15, 15, false : i64
  // CHECK: return %0 : i64
  %0 = amdgpu.dpp %arg0 %arg1 quad_perm ( [0,2,1,1] ) : i64
    return %0 : i64
}

func.func @row_bcast_dpp(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK-LABEL: func @row_bcast_dpp
  // CHECK: rocdl.update.dpp %arg0, %arg1 with 323, 4, 1, false : f64
  // CHECK: return %0 : f64
  %0 = amdgpu.dpp %arg0 %arg1 row_bcast_31 { row_mask = 0x4 : i32, bank_mask = 0x1 : i32} : f64
    return %0 : f64
}

func.func @test_dpp_f16(%arg0: f16, %arg1: f16) -> f16 {
  // CHECK-LABEL:  func @test_dpp_f16
  // CHECK: llvm.bitcast %arg1 : f16 to i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %0, %1[%2 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %3 : vector<2xi16> to i32
  // CHECK: llvm.bitcast %arg0 : f16 to i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %5, %6[%7 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %8 : vector<2xi16> to i32
  // CHECK: rocdl.update.dpp %9, %4 with 273, 15, 3, false : i32
  // CHECK: llvm.trunc %10 : i32 to i16
  // CHECK: llvm.bitcast %11 : i16 to f16
  // CHECK: return %12 : f16
  %0 = amdgpu.dpp %arg0 %arg1 row_shr ( 0x1 : i32 ){ bank_mask = 0x3 : i32 } : f16
    return %0 : f16
}

func.func @row_shl_dpp_i16(%arg0: i16, %arg1: i16) -> i16 {
  // CHECK-LABEL: func @row_shl_dpp_i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %arg1, %0[%1 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %2 : vector<2xi16> to i32
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %arg0, %4[%5 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %6 : vector<2xi16> to i32
  // CHECK: rocdl.update.dpp %7, %3 with 298, 10, 1, false : i32
  // CHECK: llvm.trunc %8 : i32 to i16
  // CHECK: return %9 : i16
  %0 = amdgpu.dpp %arg0 %arg1 row_ror ( 0xa : i32 ) { row_mask = 0xa : i32, bank_mask = 0x1 : i32 } : i16
    return %0 : i16
}

func.func @row_bcast_update_dpp_f16(%arg0: f16, %arg1: f16) -> f16 {
  // CHECK-LABEL: func @row_bcast_update_dpp_f16
  // CHECK: llvm.bitcast %arg1 : f16 to i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %0, %1[%2 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %arg0 : f16 to i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK  llvm.insertelement %5, %6[%7 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %8 : vector<2xi16> to i32
  // CHECK: rocdl.update.dpp %9, %4 with 322, 15, 15, true : i32
  // CHECK: llvm.trunc %10 : i32 to i16
  // CHECK: llvm.bitcast %11 : i16 to f16
  // CHECK: return %12 : f16
  %0 = amdgpu.dpp %arg0 %arg1 row_bcast_15 { bound_ctrl = true } : f16
    return %0 : f16
}
