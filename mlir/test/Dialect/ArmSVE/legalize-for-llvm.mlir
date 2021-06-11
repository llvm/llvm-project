// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sve" -convert-std-to-llvm | mlir-opt | FileCheck %s

func @arm_sve_sdot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.intr.sdot
  %0 = arm_sve.sdot %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_smmla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.intr.smmla
  %0 = arm_sve.smmla %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_udot(%a: !arm_sve.vector<16xi8>,
                   %b: !arm_sve.vector<16xi8>,
                   %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.intr.udot
  %0 = arm_sve.udot %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_ummla(%a: !arm_sve.vector<16xi8>,
                    %b: !arm_sve.vector<16xi8>,
                    %c: !arm_sve.vector<4xi32>)
    -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.intr.ummla
  %0 = arm_sve.ummla %c, %a, %b :
               !arm_sve.vector<16xi8> to !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi32>
}

func @arm_sve_arithi(%a: !arm_sve.vector<4xi32>,
                     %b: !arm_sve.vector<4xi32>,
                     %c: !arm_sve.vector<4xi32>,
                     %d: !arm_sve.vector<4xi32>,
                     %e: !arm_sve.vector<4xi32>) -> !arm_sve.vector<4xi32> {
  // CHECK: llvm.mul {{.*}}: !llvm.vec<? x 4 x i32>
  %0 = arm_sve.muli %a, %b : !arm_sve.vector<4xi32>
  // CHECK: llvm.add {{.*}}: !llvm.vec<? x 4 x i32>
  %1 = arm_sve.addi %0, %c : !arm_sve.vector<4xi32>
  // CHECK: llvm.sub {{.*}}: !llvm.vec<? x 4 x i32>
  %2 = arm_sve.subi %1, %d : !arm_sve.vector<4xi32>
  // CHECK: llvm.sdiv {{.*}}: !llvm.vec<? x 4 x i32>
  %3 = arm_sve.divi_signed %2, %e : !arm_sve.vector<4xi32>
  // CHECK: llvm.udiv {{.*}}: !llvm.vec<? x 4 x i32>
  %4 = arm_sve.divi_unsigned %2, %e : !arm_sve.vector<4xi32>
  return %4 : !arm_sve.vector<4xi32>
}

func @arm_sve_arithf(%a: !arm_sve.vector<4xf32>,
                     %b: !arm_sve.vector<4xf32>,
                     %c: !arm_sve.vector<4xf32>,
                     %d: !arm_sve.vector<4xf32>,
                     %e: !arm_sve.vector<4xf32>) -> !arm_sve.vector<4xf32> {
  // CHECK: llvm.fmul {{.*}}: !llvm.vec<? x 4 x f32>
  %0 = arm_sve.mulf %a, %b : !arm_sve.vector<4xf32>
  // CHECK: llvm.fadd {{.*}}: !llvm.vec<? x 4 x f32>
  %1 = arm_sve.addf %0, %c : !arm_sve.vector<4xf32>
  // CHECK: llvm.fsub {{.*}}: !llvm.vec<? x 4 x f32>
  %2 = arm_sve.subf %1, %d : !arm_sve.vector<4xf32>
  // CHECK: llvm.fdiv {{.*}}: !llvm.vec<? x 4 x f32>
  %3 = arm_sve.divf %2, %e : !arm_sve.vector<4xf32>
  return %3 : !arm_sve.vector<4xf32>
}

func @arm_sve_arithi_masked(%a: !arm_sve.vector<4xi32>,
                            %b: !arm_sve.vector<4xi32>,
                            %c: !arm_sve.vector<4xi32>,
                            %d: !arm_sve.vector<4xi32>,
                            %e: !arm_sve.vector<4xi32>,
                            %mask: !arm_sve.vector<4xi1>
                            ) -> !arm_sve.vector<4xi32> {
  // CHECK: arm_sve.intr.add{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %0 = arm_sve.masked.addi %mask, %a, %b : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.intr.sub{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %1 = arm_sve.masked.subi %mask, %0, %c : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.intr.mul{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %2 = arm_sve.masked.muli %mask, %1, %d : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xi32>
  // CHECK: arm_sve.intr.sdiv{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %3 = arm_sve.masked.divi_signed %mask, %2, %e : !arm_sve.vector<4xi1>,
                                                  !arm_sve.vector<4xi32>
  // CHECK: arm_sve.intr.udiv{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %4 = arm_sve.masked.divi_unsigned %mask, %3, %e : !arm_sve.vector<4xi1>,
                                                    !arm_sve.vector<4xi32>
  return %4 : !arm_sve.vector<4xi32>
}

func @arm_sve_arithf_masked(%a: !arm_sve.vector<4xf32>,
                            %b: !arm_sve.vector<4xf32>,
                            %c: !arm_sve.vector<4xf32>,
                            %d: !arm_sve.vector<4xf32>,
                            %e: !arm_sve.vector<4xf32>,
                            %mask: !arm_sve.vector<4xi1>
                            ) -> !arm_sve.vector<4xf32> {
  // CHECK: arm_sve.intr.fadd{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x f32>, !llvm.vec<? x 4 x f32>) -> !llvm.vec<? x 4 x f32>
  %0 = arm_sve.masked.addf %mask, %a, %b : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.intr.fsub{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x f32>, !llvm.vec<? x 4 x f32>) -> !llvm.vec<? x 4 x f32>
  %1 = arm_sve.masked.subf %mask, %0, %c : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.intr.fmul{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x f32>, !llvm.vec<? x 4 x f32>) -> !llvm.vec<? x 4 x f32>
  %2 = arm_sve.masked.mulf %mask, %1, %d : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  // CHECK: arm_sve.intr.fdiv{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x f32>, !llvm.vec<? x 4 x f32>) -> !llvm.vec<? x 4 x f32>
  %3 = arm_sve.masked.divf %mask, %2, %e : !arm_sve.vector<4xi1>,
                                           !arm_sve.vector<4xf32>
  return %3 : !arm_sve.vector<4xf32>
}

func @arm_sve_mask_genf(%a: !arm_sve.vector<4xf32>,
                        %b: !arm_sve.vector<4xf32>)
                        -> !arm_sve.vector<4xi1> {
  // CHECK: llvm.fcmp "oeq" {{.*}}: !llvm.vec<? x 4 x f32>
  %0 = arm_sve.cmpf oeq, %a, %b : !arm_sve.vector<4xf32>
  return %0 : !arm_sve.vector<4xi1>
}

func @arm_sve_mask_geni(%a: !arm_sve.vector<4xi32>,
                        %b: !arm_sve.vector<4xi32>)
                        -> !arm_sve.vector<4xi1> {
  // CHECK: llvm.icmp "uge" {{.*}}: !llvm.vec<? x 4 x i32>
  %0 = arm_sve.cmpi uge, %a, %b : !arm_sve.vector<4xi32>
  return %0 : !arm_sve.vector<4xi1>
}

func @arm_sve_abs_diff(%a: !arm_sve.vector<4xi32>,
                       %b: !arm_sve.vector<4xi32>)
                       -> !arm_sve.vector<4xi32> {
  // CHECK: llvm.sub {{.*}}: !llvm.vec<? x 4 x i32>
  %z = arm_sve.subi %a, %a : !arm_sve.vector<4xi32>
  // CHECK: llvm.icmp "sge" {{.*}}: !llvm.vec<? x 4 x i32>
  %agb = arm_sve.cmpi sge, %a, %b : !arm_sve.vector<4xi32>
  // CHECK: llvm.icmp "slt" {{.*}}: !llvm.vec<? x 4 x i32>
  %bga = arm_sve.cmpi slt, %a, %b : !arm_sve.vector<4xi32>
  // CHECK: "arm_sve.intr.sub"{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %0 = arm_sve.masked.subi %agb, %a, %b : !arm_sve.vector<4xi1>,
                                          !arm_sve.vector<4xi32>
  // CHECK: "arm_sve.intr.sub"{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %1 = arm_sve.masked.subi %bga, %b, %a : !arm_sve.vector<4xi1>,
                                          !arm_sve.vector<4xi32>
  // CHECK: "arm_sve.intr.add"{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %2 = arm_sve.masked.addi %agb, %z, %0 : !arm_sve.vector<4xi1>,
                                          !arm_sve.vector<4xi32>
  // CHECK: "arm_sve.intr.add"{{.*}}: (!llvm.vec<? x 4 x i1>, !llvm.vec<? x 4 x i32>, !llvm.vec<? x 4 x i32>) -> !llvm.vec<? x 4 x i32>
  %3 = arm_sve.masked.addi %bga, %2, %1 : !arm_sve.vector<4xi1>,
                                          !arm_sve.vector<4xi32>
  return %3 : !arm_sve.vector<4xi32>
}

func @get_vector_scale() -> index {
  // CHECK: arm_sve.vscale
  %0 = arm_sve.vector_scale : index
  return %0 : index
}
