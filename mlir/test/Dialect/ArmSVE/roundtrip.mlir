// RUN: mlir-opt -verify-diagnostics -split-input-file %s | mlir-opt | FileCheck %s

func.func @arm_sve_sdot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.sdot {{.*}}: vector<[16]xi8> to vector<[4]xi32
  %0 = arm_sve.sdot %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_smmla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.smmla {{.*}}: vector<[16]xi8> to vector<[4]xi3
  %0 = arm_sve.smmla %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_udot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.udot {{.*}}: vector<[16]xi8> to vector<[4]xi32
  %0 = arm_sve.udot %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.ummla {{.*}}: vector<[16]xi8> to vector<[4]xi3
  %0 = arm_sve.ummla %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_masked_arithi(%a: vector<[4]xi32>,
                            %b: vector<[4]xi32>,
                            %c: vector<[4]xi32>,
                            %d: vector<[4]xi32>,
                            %e: vector<[4]xi32>,
                            %mask: vector<[4]xi1>)
                            -> vector<[4]xi32> {
  // CHECK: arm_sve.masked.muli {{.*}}: vector<[4]xi1>, vector<
  %0 = arm_sve.masked.muli %mask, %a, %b : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.masked.addi {{.*}}: vector<[4]xi1>, vector<
  %1 = arm_sve.masked.addi %mask, %0, %c : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.masked.subi {{.*}}: vector<[4]xi1>, vector<
  %2 = arm_sve.masked.subi %mask, %1, %d : vector<[4]xi1>,
                                           vector<[4]xi32>
  // CHECK: arm_sve.masked.divi_signed
  %3 = arm_sve.masked.divi_signed %mask, %2, %e : vector<[4]xi1>,
                                                  vector<[4]xi32>
  // CHECK: arm_sve.masked.divi_unsigned
  %4 = arm_sve.masked.divi_unsigned %mask, %3, %e : vector<[4]xi1>,
                                                    vector<[4]xi32>
  return %2 : vector<[4]xi32>
}

// -----

func.func @arm_sve_masked_arithf(%a: vector<[4]xf32>,
                            %b: vector<[4]xf32>,
                            %c: vector<[4]xf32>,
                            %d: vector<[4]xf32>,
                            %e: vector<[4]xf32>,
                            %mask: vector<[4]xi1>)
                            -> vector<[4]xf32> {
  // CHECK: arm_sve.masked.mulf {{.*}}: vector<[4]xi1>, vector<
  %0 = arm_sve.masked.mulf %mask, %a, %b : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.masked.addf {{.*}}: vector<[4]xi1>, vector<
  %1 = arm_sve.masked.addf %mask, %0, %c : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.masked.subf {{.*}}: vector<[4]xi1>, vector<
  %2 = arm_sve.masked.subf %mask, %1, %d : vector<[4]xi1>,
                                           vector<[4]xf32>
  // CHECK: arm_sve.masked.divf {{.*}}: vector<[4]xi1>, vector<
  %3 = arm_sve.masked.divf %mask, %2, %e : vector<[4]xi1>,
                                           vector<[4]xf32>
  return %3 : vector<[4]xf32>
}

// -----

func.func @arm_sve_convert_to_svbool(%a: vector<[1]xi1>,
                                     %b: vector<[2]xi1>,
                                     %c: vector<[4]xi1>,
                                     %d: vector<[8]xi1>) {
  // CHECK: arm_sve.convert_to_svbool %{{.*}} : vector<[1]xi1>
  %1 = arm_sve.convert_to_svbool %a : vector<[1]xi1>

  // CHECK: arm_sve.convert_to_svbool %{{.*}} : vector<[2]xi1>
  %2 = arm_sve.convert_to_svbool %b : vector<[2]xi1>

  // CHECK: arm_sve.convert_to_svbool %{{.*}} : vector<[4]xi1>
  %3 = arm_sve.convert_to_svbool %c : vector<[4]xi1>

  // CHECK: arm_sve.convert_to_svbool %{{.*}} : vector<[8]xi1>
  %4 = arm_sve.convert_to_svbool %d : vector<[8]xi1>
  return
}

// -----

func.func @arm_sve_convert_from_svbool(%bool: vector<[16]xi1>) {
  // CHECK: arm_sve.convert_from_svbool %{{.*}} : vector<[1]xi1>
  %1 = arm_sve.convert_from_svbool %bool : vector<[1]xi1>

  // CHECK: arm_sve.convert_from_svbool %{{.*}} : vector<[2]xi1>
  %2 = arm_sve.convert_from_svbool %bool : vector<[2]xi1>

  // CHECK: arm_sve.convert_from_svbool %{{.*}} : vector<[4]xi1>
  %3 = arm_sve.convert_from_svbool %bool : vector<[4]xi1>

  // CHECK: arm_sve.convert_from_svbool %{{.*}} : vector<[8]xi1>
  %4 = arm_sve.convert_from_svbool %bool : vector<[8]xi1>
  return
}
