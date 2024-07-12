// RUN: mlir-opt --split-input-file --arith-emulate-unsupported-floats="source-types=bf16,f8E4M3FNUZ target-type=f32" %s | FileCheck %s

func.func @basic_expansion(%x: bf16) -> bf16 {
// CHECK-LABEL: @basic_expansion
// CHECK-SAME: [[X:%.+]]: bf16
// CHECK-DAG: [[C:%.+]] = arith.constant {{.*}} : bf16
// CHECK-DAG: [[X_EXP:%.+]] = arith.extf [[X]] fastmath<contract> : bf16 to f32
// CHECK-DAG: [[C_EXP:%.+]] = arith.extf [[C]] fastmath<contract> : bf16 to f32
// CHECK: [[Y_EXP:%.+]] = arith.addf [[X_EXP]], [[C_EXP]] : f32
// CHECK: [[Y:%.+]] = arith.truncf [[Y_EXP]] fastmath<contract> : f32 to bf16
// CHECK: return [[Y]]
  %c = arith.constant 1.0 : bf16
  %y = arith.addf %x, %c : bf16
  func.return %y : bf16
}

// -----

func.func @chained(%x: bf16, %y: bf16, %z: bf16) -> i1 {
// CHECK-LABEL: @chained
// CHECK-SAME: [[X:%.+]]: bf16, [[Y:%.+]]: bf16, [[Z:%.+]]: bf16
// CHECK-DAG: [[X_EXP:%.+]] = arith.extf [[X]] fastmath<contract> : bf16 to f32
// CHECK-DAG: [[Y_EXP:%.+]] = arith.extf [[Y]] fastmath<contract> : bf16 to f32
// CHECK-DAG: [[Z_EXP:%.+]] = arith.extf [[Z]] fastmath<contract> : bf16 to f32
// CHECK: [[P_EXP:%.+]] = arith.addf [[X_EXP]], [[Y_EXP]] : f32
// CHECK: [[P:%.+]] = arith.truncf [[P_EXP]] fastmath<contract> : f32 to bf16
// CHECK: [[P_EXP2:%.+]] = arith.extf [[P]] fastmath<contract> : bf16 to f32
// CHECK: [[Q_EXP:%.+]] = arith.mulf [[P_EXP2]], [[Z_EXP]]
// CHECK: [[Q:%.+]] = arith.truncf [[Q_EXP]] fastmath<contract> : f32 to bf16
// CHECK: [[Q_EXP2:%.+]] = arith.extf [[Q]] fastmath<contract> : bf16 to f32
// CHECK: [[RES:%.+]] = arith.cmpf ole, [[P_EXP2]], [[Q_EXP2]] : f32
// CHECK: return [[RES]]
  %p = arith.addf %x, %y : bf16
  %q = arith.mulf %p, %z : bf16
  %res = arith.cmpf ole, %p, %q : bf16
  func.return %res : i1
}

// -----

func.func @memops(%a: memref<4xf8E4M3FNUZ>, %b: memref<4xf8E4M3FNUZ>) {
// CHECK-LABEL: @memops
// CHECK: [[V:%.+]] = memref.load {{.*}} : memref<4xf8E4M3FNUZ>
// CHECK: [[V_EXP:%.+]] = arith.extf [[V]] fastmath<contract> : f8E4M3FNUZ to f32
// CHECK: memref.store [[V]]
// CHECK: [[W:%.+]] = memref.load
// CHECK: [[W_EXP:%.+]] = arith.extf [[W]] fastmath<contract> : f8E4M3FNUZ to f32
// CHECK: [[X_EXP:%.+]] = arith.addf [[V_EXP]], [[W_EXP]] : f32
// CHECK: [[X:%.+]] = arith.truncf [[X_EXP]] fastmath<contract> : f32 to f8E4M3FNUZ
// CHECK: memref.store [[X]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = memref.load %a[%c0] : memref<4xf8E4M3FNUZ>
  memref.store %v, %b[%c0] : memref<4xf8E4M3FNUZ>
  %w = memref.load %a[%c1] : memref<4xf8E4M3FNUZ>
  %x = arith.addf %v, %w : f8E4M3FNUZ
  memref.store %x, %b[%c1] : memref<4xf8E4M3FNUZ>
  func.return
}

// -----

func.func @vectors(%a: vector<4xf8E4M3FNUZ>) -> vector<4xf32> {
// CHECK-LABEL: @vectors
// CHECK-SAME: [[A:%.+]]: vector<4xf8E4M3FNUZ>
// CHECK: [[A_EXP:%.+]] = arith.extf [[A]] fastmath<contract> : vector<4xf8E4M3FNUZ> to vector<4xf32>
// CHECK: [[B_EXP:%.+]] = arith.mulf [[A_EXP]], [[A_EXP]] : vector<4xf32>
// CHECK: [[B:%.+]] = arith.truncf [[B_EXP]] fastmath<contract> : vector<4xf32> to vector<4xf8E4M3FNUZ>
// CHECK: [[RET:%.+]] = arith.extf [[B]] : vector<4xf8E4M3FNUZ> to vector<4xf32>
// CHECK: return [[RET]]
  %b = arith.mulf %a, %a : vector<4xf8E4M3FNUZ>
  %ret = arith.extf %b : vector<4xf8E4M3FNUZ> to vector<4xf32>
  func.return %ret : vector<4xf32>
}

// -----

func.func @no_expansion(%x: f32) -> f32 {
// CHECK-LABEL: @no_expansion
// CHECK-SAME: [[X:%.+]]: f32
// CHECK-DAG: [[C:%.+]] = arith.constant {{.*}} : f32
// CHECK: [[Y:%.+]] = arith.addf [[X]], [[C]] : f32
// CHECK: return [[Y]]
  %c = arith.constant 1.0 : f32
  %y = arith.addf %x, %c : f32
  func.return %y : f32
}
