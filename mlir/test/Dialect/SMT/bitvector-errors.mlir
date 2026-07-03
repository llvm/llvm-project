// RUN: mlir-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{bit-vector must have at least a width of one}}
func.func @at_least_size_one(%arg0: !smt.bv<0>) {
  return
}

// -----

// expected-error @below {{bit-vector must have at least a width of one}}
func.func @positive_width(%arg0: !smt.bv<-1>) {
  return
}

// -----

func.func @attr_type_and_return_type_match() {
  // expected-error @below {{inferred type(s) '!smt.bv<1>' are incompatible with return type(s) of operation '!smt.bv<32>'}}
  // expected-error @below {{failed to infer returned types}}
  %c0_bv32 = "smt.bv.constant"() <{value = #smt.bv<0> : !smt.bv<1>}> : () -> !smt.bv<32>
  return
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{explicit bit-vector type required}}
  smt.bv.constant #smt.bv<5>
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{integer value out of range for given bit-vector type}}
  smt.bv.constant #smt.bv<32> : !smt.bv<2>
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{integer value out of range for given bit-vector type}}
  smt.bv.constant #smt.bv<-4> : !smt.bv<2>
}

// -----

func.func @extraction(%arg0: !smt.bv<32>) {
  // expected-error @below {{range to be extracted is too big, expected range starting at index 20 of length 16 requires input width of at least 36, but the input width is only 32}}
  smt.bv.extract %arg0 from 20 : (!smt.bv<32>) -> !smt.bv<16>
  return
}

// -----

func.func @concat(%arg0: !smt.bv<32>) {
  // expected-error @below {{inferred type(s) '!smt.bv<64>' are incompatible with return type(s) of operation '!smt.bv<33>'}}
  // expected-error @below {{failed to infer returned types}}
  "smt.bv.concat"(%arg0, %arg0) {} : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<33>
  return
}

// -----

func.func @repeat_result_type_no_multiple_of_input_type(%arg0: !smt.bv<32>) {
  // expected-error @below {{result bit-vector width must be a multiple of the input bit-vector width}}
  "smt.bv.repeat"(%arg0) : (!smt.bv<32>) -> !smt.bv<65>
  return
}

// -----

func.func @repeat_negative_count(%arg0: !smt.bv<32>) {
  // expected-error @below {{integer must be positive}}
  smt.bv.repeat -2 times %arg0 : !smt.bv<32>
  return
}

// -----

// The parser has to extract the bit-width of the input and thus we need to
// test that this is handled correctly in the parser, we cannot just rely on the
// verifier.
func.func @repeat_wrong_input_type(%arg0: !smt.bool) {
  // expected-error @below {{input must have bit-vector type}}
  smt.bv.repeat 2 times %arg0 : !smt.bool
  return
}

// -----

func.func @repeat_count_too_large(%arg0: !smt.bv<32>) {
  // expected-error @below {{integer must fit into 63 bits}}
  smt.bv.repeat 18446744073709551617 times %arg0 : !smt.bv<32>
  return
}

// -----

func.func @repeat_result_type_bitwidth_too_large(%arg0: !smt.bv<9223372036854775807>) {
  // expected-error @below {{result bit-width (provided integer times bit-width of the input type) must fit into 63 bits}}
  smt.bv.repeat 2 times %arg0 : !smt.bv<9223372036854775807>
  return
}

// -----

func.func @invalid_bv2int_signedness() {
  %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32>
  // expected-error @below {{expected ':'}}
  %bv2int = smt.bv2int %c5_bv32 unsigned : !smt.bv<32>
  return
}
