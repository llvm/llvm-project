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


