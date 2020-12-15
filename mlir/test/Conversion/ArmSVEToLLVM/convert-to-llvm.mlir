// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sve" | mlir-opt | FileCheck %s

func @arm_sve_sdot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: llvm_arm_sve.sdot
  %0 = arm_sve.sdot %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_smmla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: llvm_arm_sve.smmla
  %0 = arm_sve.smmla %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_udot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: llvm_arm_sve.udot
  %0 = arm_sve.udot %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_ummla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: llvm_arm_sve.ummla
  %0 = arm_sve.ummla %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @get_vector_scale() -> index {
  // CHECK: llvm_arm_sve.vscale
  %0 = arm_sve.vector_scale : index
  return %0 : index
}
