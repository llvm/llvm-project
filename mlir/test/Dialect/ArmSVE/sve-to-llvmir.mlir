// RUN: mlir-opt -convert-vector-to-llvm="enable-arm-sve" -convert-arith-to-llvm -convert-func-to-llvm %s --split-input-file | bin/mlir-translate --mlir-to-llvmir | FileCheck %s

func.func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: call {{.*}} @llvm.aarch64.sve.ummla.nxv4i32
  %0 = arm_sve.ummla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_fmla(%a: vector<[4]xf32>,
                    %b: vector<[4]xf32>,
                    %c: vector<[4]xf32>,
                    %d: f32)
    -> vector<[4]xf32> {
  // CHECK: call {{.*}} @llvm.aarch64.sve.fmla.lane.nxv4f32
  %c3 = arith.constant 3 : i32
  %0 = arm_sve.fmla %c, %a, %b, %c3 :
               vector<[4]xf32> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}
