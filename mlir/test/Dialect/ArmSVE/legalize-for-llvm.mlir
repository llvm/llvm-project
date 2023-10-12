// RUN: mlir-opt -convert-vector-to-llvm="enable-arm-sve" -convert-func-to-llvm -reconcile-unrealized-casts -split-input-file %s | FileCheck %s

func.func @arm_sve_sdot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.sdot
  %0 = arm_sve.sdot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_smmla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.smmla
  %0 = arm_sve.smmla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_udot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.udot
  %0 = arm_sve.udot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.ummla
  %0 = arm_sve.ummla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @arm_sve_arithi_masked(%a: vector<[4]xi32>,
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

// -----

func.func @arm_sve_arithf_masked(%a: vector<[4]xf32>,
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

// -----

func.func @arm_sve_abs_diff(%a: vector<[4]xi32>,
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

// -----

func.func @get_vector_scale() -> index {
  // CHECK: llvm.intr.vscale
  %0 = vector.vscale
  return %0 : index
}

// -----

func.func @convert_1d_mask_to_svbool(%mask: vector<[4]xi1>) -> vector<[16]xi1>
{
  // CHECK: "arm_sve.intr.convert.to.svbool"(%{{.*}}) : (vector<[4]xi1>) -> vector<[16]xi1>
  %svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
  return %svbool : vector<[16]xi1>
}

// -----

func.func @convert_1d_mask_from_svbool(%svbool: vector<[16]xi1>) -> vector<[2]xi1>
{
  // CHECK: "arm_sve.intr.convert.from.svbool"(%{{.*}}) : (vector<[16]xi1>) -> vector<[2]xi1>
  %mask = arm_sve.convert_from_svbool %svbool : vector<[2]xi1>
  return %mask : vector<[2]xi1>
}

// -----

// CHECK-LABEL: @convert_2d_mask_to_svbool(
// CHECK-SAME:                             %[[MASK:.*]]: !llvm.array<2 x vector<[8]xi1>>)
func.func @convert_2d_mask_to_svbool(%mask: vector<2x[8]xi1>) -> vector<2x[16]xi1>
{
  // CHECK-NEXT:    %[[RES0:.*]] = llvm.mlir.constant(dense<false> : vector<2x[16]xi1>) : !llvm.array<2 x vector<[16]xi1>>
  // CHECK-NEXT:   %[[MASK0:.*]] = llvm.extractvalue %[[MASK]][0] : !llvm.array<2 x vector<[8]xi1>>
  // CHECK-NEXT: %[[SVBOOL0:.*]] = "arm_sve.intr.convert.to.svbool"(%[[MASK0]]) : (vector<[8]xi1>) -> vector<[16]xi1>
  // CHECK-NEXT:    %[[RES1:.*]] = llvm.insertvalue %[[SVBOOL0]], %[[RES0]][0] : !llvm.array<2 x vector<[16]xi1>>
  // CHECK-NEXT:   %[[MASK1:.*]] = llvm.extractvalue %[[MASK]][1] : !llvm.array<2 x vector<[8]xi1>>
  // CHECK-NEXT: %[[SVBOOL1:.*]] = "arm_sve.intr.convert.to.svbool"(%[[MASK1]]) : (vector<[8]xi1>) -> vector<[16]xi1>
  // CHECK-NEXT:  %[[SVBOOL:.*]] = llvm.insertvalue %[[SVBOOL1]], %[[RES1]][1] : !llvm.array<2 x vector<[16]xi1>>
  %svbool = arm_sve.convert_to_svbool %mask : vector<2x[8]xi1>
  // CHECK-NEXT: llvm.return %[[SVBOOL]] : !llvm.array<2 x vector<[16]xi1>>
  return %svbool : vector<2x[16]xi1>
}

// -----

// CHECK-LABEL: @convert_2d_mask_from_svbool(
// CHECK-SAME:                               %[[SVBOOL:.*]]: !llvm.array<3 x vector<[16]xi1>>)
func.func @convert_2d_mask_from_svbool(%svbool: vector<3x[16]xi1>) -> vector<3x[1]xi1>
{
  // CHECK-NEXT:    %[[RES0:.*]] = llvm.mlir.constant(dense<false> : vector<3x[1]xi1>) : !llvm.array<3 x vector<[1]xi1>>
  // CHECK-NEXT: %[[SVBOOL0:.*]] = llvm.extractvalue %[[SVBOOL]][0] : !llvm.array<3 x vector<[16]xi1>>
  // CHECK-NEXT:   %[[MASK0:.*]] = "arm_sve.intr.convert.from.svbool"(%[[SVBOOL0]]) : (vector<[16]xi1>) -> vector<[1]xi1>
  // CHECK-NEXT:    %[[RES1:.*]] = llvm.insertvalue %[[MASK0]], %[[RES0]][0] : !llvm.array<3 x vector<[1]xi1>>
  // CHECK-NEXT: %[[SVBOOL1:.*]] = llvm.extractvalue %[[SVBOOL]][1] : !llvm.array<3 x vector<[16]xi1>>
  // CHECK-NEXT:   %[[MASK1:.*]] = "arm_sve.intr.convert.from.svbool"(%[[SVBOOL1]]) : (vector<[16]xi1>) -> vector<[1]xi1>
  // CHECK-NEXT:    %[[RES2:.*]] = llvm.insertvalue %[[MASK1]], %[[RES1]][1] : !llvm.array<3 x vector<[1]xi1>>
  // CHECK-NEXT: %[[SVBOOL2:.*]] = llvm.extractvalue %[[SVBOOL]][2] : !llvm.array<3 x vector<[16]xi1>>
  // CHECK-NEXT:   %[[MASK2:.*]] = "arm_sve.intr.convert.from.svbool"(%[[SVBOOL2]]) : (vector<[16]xi1>) -> vector<[1]xi1>
  // CHECK-NEXT:    %[[MASK:.*]] = llvm.insertvalue %[[MASK2]], %[[RES2]][2] : !llvm.array<3 x vector<[1]xi1>>
  %mask = arm_sve.convert_from_svbool %svbool : vector<3x[1]xi1>
  // CHECK-NEXT: llvm.return %[[MASK]] : !llvm.array<3 x vector<[1]xi1>>
  return %mask : vector<3x[1]xi1>
}
