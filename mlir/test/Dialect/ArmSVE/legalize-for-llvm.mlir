// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sve" -convert-func-to-llvm -reconcile-unrealized-casts | mlir-opt | FileCheck %s

func @arm_sve_sdot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.sdot
  %0 = arm_sve.sdot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func @arm_sve_smmla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.smmla
  %0 = arm_sve.smmla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func @arm_sve_udot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.udot
  %0 = arm_sve.udot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.ummla
  %0 = arm_sve.ummla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func @arm_sve_arithi_masked(%a: vector<[4]xi32>,
                            %b: vector<[4]xi32>,
                            %c: vector<[4]xi32>,
                            %d: vector<[4]xi32>,
                            %e: vector<[4]xi32>,
                            %mask: vector<[4]xi1>
                            ) -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.add{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %0 = arm_sve.masked.addi %mask, %a, %b : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.intr.sub{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %1 = arm_sve.masked.subi %mask, %0, %c : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.intr.mul{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %2 = arm_sve.masked.muli %mask, %1, %d : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.intr.sdiv{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %3 = arm_sve.masked.divi_signed %mask, %2, %e : vector<[4]xi1>,
                                                  vector<[4]xi32>
  // CHECK: arm_sve.intr.udiv{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %4 = arm_sve.masked.divi_unsigned %mask, %3, %e : vector<[4]xi1>,
                                                    vector<[4]xi32>
  return %4 : vector<[4]xi32>
}

func @arm_sve_arithf_masked(%a: vector<[4]xf32>,
                            %b: vector<[4]xf32>,
                            %c: vector<[4]xf32>,
                            %d: vector<[4]xf32>,
                            %e: vector<[4]xf32>,
                            %mask: vector<[4]xi1>
                            ) -> vector<[4]xf32> {
  // CHECK: arm_sve.intr.fadd{{.*}}: (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  %0 = arm_sve.masked.addf %mask, %a, %b : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.intr.fsub{{.*}}: (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  %1 = arm_sve.masked.subf %mask, %0, %c : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.intr.fmul{{.*}}: (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  %2 = arm_sve.masked.mulf %mask, %1, %d : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.intr.fdiv{{.*}}: (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
  %3 = arm_sve.masked.divf %mask, %2, %e : vector<[4]xi1>,
                                           vector<[4]xf32>
  return %3 : vector<[4]xf32>
}

func @arm_sve_abs_diff(%a: vector<[4]xi32>,
                       %b: vector<[4]xi32>)
                       -> vector<[4]xi32> {
  // CHECK: llvm.mlir.constant(dense<0> : vector<[4]xi32>) : vector<[4]xi32>
  %z = arith.subi %a, %a : vector<[4]xi32>
  // CHECK: llvm.icmp "sge" {{.*}}: vector<[4]xi32>
  %agb = arith.cmpi sge, %a, %b : vector<[4]xi32>
  // CHECK: llvm.icmp "slt" {{.*}}: vector<[4]xi32>
  %bga = arith.cmpi slt, %a, %b : vector<[4]xi32>
  // CHECK: "arm_sve.intr.sub"{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %0 = arm_sve.masked.subi %agb, %a, %b : vector<[4]xi1>,
                                          vector<[4]xi32>
  // CHECK: "arm_sve.intr.sub"{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %1 = arm_sve.masked.subi %bga, %b, %a : vector<[4]xi1>,
                                          vector<[4]xi32>
  // CHECK: "arm_sve.intr.add"{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %2 = arm_sve.masked.addi %agb, %z, %0 : vector<[4]xi1>,
                                          vector<[4]xi32>
  // CHECK: "arm_sve.intr.add"{{.*}}: (vector<[4]xi1>, vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
  %3 = arm_sve.masked.addi %bga, %2, %1 : vector<[4]xi1>,
                                          vector<[4]xi32>
  return %3 : vector<[4]xi32>
}

func @get_vector_scale() -> index {
  // CHECK: llvm.intr.vscale
  %0 = vector.vscale
  return %0 : index
}
