// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

func @arm_sve_sdot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.sdot {{.*}}: !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32
  %0 = arm_sve.sdot %c, %a, %b :
             !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_smmla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.smmla {{.*}}: !arm_sve.vector<16xi8> to !arm_sve.vector<4xi3
  %0 = arm_sve.smmla %c, %a, %b :
             !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_udot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.udot {{.*}}: !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32
  %0 = arm_sve.udot %c, %a, %b :
             !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_ummla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.ummla {{.*}}: !arm_sve.vector<16xi8> to !arm_sve.vector<4xi3
  %0 = arm_sve.ummla %c, %a, %b :
             !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_arithi(%a: !arm_sve.vector<4xi32>,
                     %b: !arm_sve.vector<4xi32>,
                     %c: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.muli {{.*}}: !arm_sve.vector<4xi32>
  %0 = arm_sve.muli %a, %b : !arm_sve.vector<4xi32>
  // CHECK: arm_sve.addi {{.*}}: !arm_sve.vector<4xi32>
  %1 = arm_sve.addi %0, %c : !arm_sve.vector<4xi32>
  return %1 : !arm_sve.vector<4xi32>
}

func @arm_sve_arithf(%a: !arm_sve.vector<4xf32>,
                     %b: !arm_sve.vector<4xf32>,
                     %c: !arm_sve.vector<4xf32>) -> !arm_sve.vector<4xf32> {
  // CHECK: arm_sve.mulf {{.*}}: !arm_sve.vector<4xf32>
  %0 = arm_sve.mulf %a, %b : !arm_sve.vector<4xf32>
  // CHECK: arm_sve.addf {{.*}}: !arm_sve.vector<4xf32>
  %1 = arm_sve.addf %0, %c : !arm_sve.vector<4xf32>
  return %1 : !arm_sve.vector<4xf32>
}

func @get_vector_scale() -> index {
  // CHECK: arm_sve.vector_scale : index
  %0 = arm_sve.vector_scale : index
  return %0 : index
}
