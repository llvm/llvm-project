// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @arm_sve_convert_from_svbool__bad_mask_type(%bool: vector<2x[16]xi1>) -> vector<2x[8]xi2> {
  // expected-error@+1 {{'result' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<2x[8]xi2>'}}
  %mask = arm_sve.convert_from_svbool %bool : vector<2x[8]xi2>
  return %mask : vector<2x[8]xi2>
}

// -----

func.func @arm_sve_convert_from_svbool__bad_mask_shape(%bool : vector<[16]xi1>) -> vector<[7]xi1> {
  // expected-error@+1 {{'result' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<[7]xi1>'}}
  %mask = arm_sve.convert_from_svbool %bool : vector<[7]xi1>
  return %mask : vector<[7]xi1>
}

// -----

func.func @arm_sve_convert_from_svbool__bad_mask_scalability(%bool : vector<[4]x[16]xi1>) -> vector<[4]x[8]xi1> {
  // expected-error@+1 {{'result' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<[4]x[8]xi1>'}}
  %mask = arm_sve.convert_from_svbool %bool : vector<[4]x[8]xi1>
  return %mask : vector<[4]x[8]xi1>
}

// -----

func.func @arm_sve_convert_to_svbool__bad_mask_type(%mask: vector<2x[8]xi2>) -> vector<2x[16]xi1> {
  // expected-error@+1 {{'source' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<2x[8]xi2>'}}
  %bool = arm_sve.convert_to_svbool %mask : vector<2x[8]xi2>
  return %bool : vector<2x[16]xi1>
}

// -----

func.func @arm_sve_convert_to_svbool__bad_mask_shape(%mask : vector<[7]xi1>) -> vector<[16]xi1> {
  // expected-error@+1 {{'source' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<[7]xi1>'}}
  %bool = arm_sve.convert_to_svbool %mask : vector<[7]xi1>
  return
}

// -----

func.func @arm_sve_convert_to_svbool__bad_mask_scalability(%mask : vector<[4]x[8]xi1>) -> vector<[4]x[16]xi1> {
  // expected-error@+1 {{'source' must be trailing scalable vector of 1-bit signless integer values with dim -1 having a size of {16, 8, 4, 2, 1}, but got 'vector<[4]x[8]xi1>'}}
  %bool = arm_sve.convert_to_svbool %mask : vector<[4]x[8]xi1>
  return
}


// -----

func.func @arm_sve_zip_x2_bad_vector_type(%a : vector<[7]xi8>) {
  // expected-error@+1 {{op operand #0 must be an SVE vector with element size <= 64-bit, but got 'vector<[7]xi8>'}}
  arm_sve.zip.x2 %a, %a : vector<[7]xi8>
  return
}

// -----

func.func @arm_sve_zip_x4_bad_vector_type(%a : vector<[5]xf64>) {
  // expected-error@+1 {{op operand #0 must be an SVE vector with element size <= 64-bit, but got 'vector<[5]xf64>'}}
  arm_sve.zip.x4 %a, %a, %a, %a : vector<[5]xf64>
  return
}

// -----

func.func @arm_sve_psel_bad_vector_type(%a : vector<[7]xi1>, %index: index) {
  // expected-error@+1 {{op operand #0 must be vector<[1]xi1>, vector<[2]xi1>, vector<[4]xi1>, vector<[8]xi1>, or vector<[16]xi1>, but got 'vector<[7]xi1>'}}
  arm_sve.psel %a, %a[%index] : vector<[7]xi1>, vector<[7]xi1>
  return
}

// -----

func.func @bfmmla_invalid_element_type_lhs_rhs(%acc: vector<[4]xf32>,
                                               %lhs: vector<[8]xf16>,
                                               %rhs: vector<[8]xf16>) -> vector<[4]xf32> {
  // expected-error@+1 {{operand #1 must be scalable vector of bfloat16 type values of length 8, but got 'vector<[8]xf16>'}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<[8]xf16> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// -----

func.func @bfmmla_invalid_dimension_lhs_rhs(%acc: vector<[4]xf32>,
                                            %lhs: vector<[4]xbf16>,
                                            %rhs: vector<[4]xbf16>) -> vector<[4]xf32> {
  // expected-error@+1 {{operand #1 must be scalable vector of bfloat16 type values of length 8, but got 'vector<[4]xbf16>}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<[4]xbf16> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// -----

func.func @bfmmla_fixed_dimension_lhs_rhs(%acc: vector<[4]xf32>,
                                          %lhs: vector<8xbf16>,
                                          %rhs: vector<8xbf16>) -> vector<[4]xf32> {
  // expected-error@+1 {{operand #1 must be scalable vector of bfloat16 type values of length 8, but got 'vector<8xbf16>}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<8xbf16> to vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// -----

func.func @bfmmla_invalid_element_type_acc(%acc: vector<[4]xi32>,
                                           %lhs: vector<[8]xbf16>,
                                           %rhs: vector<[8]xbf16>) -> vector<[4]xi32> {
  // expected-error@+1 {{operand #0 must be scalable vector of 32-bit float values of length 4, but got 'vector<[4]xi32>'}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<[8]xbf16> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

// -----

func.func @bfmmla_invalid_dimension_acc(%acc: vector<[8]xf32>,
                                        %lhs: vector<[8]xbf16>,
                                        %rhs: vector<[8]xbf16>) -> vector<[8]xf32> {
  // expected-error@+1 {{operand #0 must be scalable vector of 32-bit float values of length 4, but got 'vector<[8]xf32>'}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<[8]xbf16> to vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// -----

func.func @bfmmla_fixed_dimension_acc(%acc: vector<4xf32>,
                                      %lhs: vector<[8]xbf16>,
                                      %rhs: vector<[8]xbf16>) -> vector<4xf32> {
  // expected-error@+1 {{operand #0 must be scalable vector of 32-bit float values of length 4, but got 'vector<4xf32>'}}
  %0 = arm_sve.intr.bfmmla %acc, %lhs, %rhs : vector<[8]xbf16> to vector<4xf32>
  return %0 : vector<4xf32>
}
