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

func @arm_sve_masked_arithi(%a: !arm_sve.vector<4xi32>,
                            %b: !arm_sve.vector<4xi32>,
                            %c: !arm_sve.vector<4xi32>,
                            %d: !arm_sve.vector<4xi32>,
                            %e: !arm_sve.vector<4xi32>,
                            %mask: !arm_sve.vector<4xi1>)
                            -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.masked.muli {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %0 = arm_sve.masked.muli %mask, %a, %b : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.masked.addi {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %1 = arm_sve.masked.addi %mask, %0, %c : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.masked.subi {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %2 = arm_sve.masked.subi %mask, %1, %d : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.masked.divi_signed
  %3 = arm_sve.masked.divi_signed %mask, %2, %e : !arm_sve.vector<4xi1>,
                                                  !arm_sve.vector<4xi32>
  // CHECK: arm_sve.masked.divi_unsigned
  %4 = arm_sve.masked.divi_unsigned %mask, %3, %e : !arm_sve.vector<4xi1>,
                                                    !arm_sve.vector<4xi32>
  return %2 : !arm_sve.vector<4xi32>
}

func @arm_sve_masked_arithf(%a: !arm_sve.vector<4xf32>,
                            %b: !arm_sve.vector<4xf32>,
                            %c: !arm_sve.vector<4xf32>,
                            %d: !arm_sve.vector<4xf32>,
                            %e: !arm_sve.vector<4xf32>,
                            %mask: !arm_sve.vector<4xi1>)
                            -> !arm_sve.vector<4xf32> {
  // CHECK: arm_sve.masked.mulf {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %0 = arm_sve.masked.mulf %mask, %a, %b : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.masked.addf {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %1 = arm_sve.masked.addf %mask, %0, %c : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.masked.subf {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %2 = arm_sve.masked.subf %mask, %1, %d : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.masked.divf {{.*}}: !arm_sve.vector<4xi1>, !arm_sve.vector
  %3 = arm_sve.masked.divf %mask, %2, %e : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  return %3 : !arm_sve.vector<4xf32>
}

func @arm_sve_mask_genf(%a: !arm_sve.vector<4xf32>,
                        %b: !arm_sve.vector<4xf32>)
                        -> !arm_sve.vector<4xi1> {
  // CHECK: arm_sve.cmpf oeq, {{.*}}: !arm_sve.vector<4xf32>
  %0 = arm_sve.cmpf oeq, %a, %b : !arm_sve.vector<4xf32>
  return %0 : !arm_sve.vector<4xi1>
}

func @arm_sve_mask_geni(%a: !arm_sve.vector<4xi32>,
                        %b: !arm_sve.vector<4xi32>)
                        -> !arm_sve.vector<4xi1> {
  // CHECK: arm_sve.cmpi uge, {{.*}}: !arm_sve.vector<4xi32>
  %0 = arm_sve.cmpi uge, %a, %b : !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi1>
}

func @arm_sve_memory(%v: !arm_sve.vector<4xi32>,
                     %m: memref<?xi32>)
                     -> !arm_sve.vector<4xi32> {
  %c0 = arith.constant 0 : index
  // CHECK: arm_sve.load {{.*}}: !arm_sve.vector<4xi32> from memref<?xi32>
  %0 = arm_sve.load %m[%c0] : !arm_sve.vector<4xi32> from memref<?xi32>
  // CHECK: arm_sve.store {{.*}}: !arm_sve.vector<4xi32> to memref<?xi32>
  arm_sve.store %v, %m[%c0] : !arm_sve.vector<4xi32> to memref<?xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @get_vector_scale() -> index {
  // CHECK: arm_sve.vector_scale : index
  %0 = arm_sve.vector_scale : index
  return %0 : index
}
