// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// See http://llvm.org/pr52045
// UNSUPPORTED: asan

// Check different error cases.
// -----

func.func @illegaltype(i) // expected-error {{expected non-function type}}

// -----

func.func @illegaltype() {
  %0 = arith.constant dense<0> : <vector 4 x f32> : vector<4 x f32> // expected-error {{expected non-function type}}
}

// -----

func.func @nestedtensor(tensor<tensor<i8>>) -> () // expected-error {{invalid tensor element type}}

// -----

// expected-error@+1 {{invalid dimension}}
#large_dim = tensor<9223372036854775808xf32>

// -----

func.func @illegalmemrefelementtype(memref<?xtensor<i8>>) -> () // expected-error {{invalid memref element type}}

// -----

func.func @illegalunrankedmemrefelementtype(memref<*xtensor<i8>>) -> () // expected-error {{invalid memref element type}}

// -----
// Test no map in memref type.
func.func @memrefs(memref<2x4xi8, >) // expected-error {{expected list element}}

// -----
// Test non-existent map in memref type.
func.func @memrefs(memref<2x4xi8, #map7>) // expected-error {{undefined symbol alias id 'map7'}}

// -----
// Test unsupported memory space.
func.func @memrefs(memref<2x4xi8, i8>) // expected-error {{unsupported memory space Attribute}}

// -----
// Test non-existent map in map composition of memref type.
#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @memrefs(memref<2x4xi8, #map0, #map8>) // expected-error {{undefined symbol alias id 'map8'}}

// -----
// Test multiple memory space error.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @memrefs(memref<2x4xi8, #map0, 1, 2>) // expected-error {{multiple memory spaces specified in memref type}}

// -----
// Test affine map after memory space.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @memrefs(memref<2x4xi8, #map0, 1, #map1>) // expected-error {{expected memory space to be last in memref type}}

// -----
// Test dimension mismatch between memref and layout map.
// The error must be emitted even for the trivial identity layout maps that are
// dropped in type creation.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @memrefs(memref<42xi8, #map0>) // expected-error {{memref layout mismatch between rank and affine map: 1 != 2}}

// -----

func.func @memref_space_after_strides(memref<42x42xi8, 0, offset: ?, strides: [?, ?]>) // expected-error {{expected memory space to be last in memref type}}

// -----

func.func @memref_stride_missing_colon(memref<42x42xi8, offset ?, strides: [?, ?]>) // expected-error {{expected colon after `offset` keyword}}

// -----

func.func @memref_stride_invalid_offset(memref<42x42xi8, offset: [], strides: [?, ?]>) // expected-error {{invalid offset}}

// -----

func.func @memref_stride_missing_strides(memref<42x42xi8, offset: 0 [?, ?]>) // expected-error {{expected comma after offset value}}

// -----

func.func @memref_stride_missing_strides(memref<42x42xi8, offset: 0, [?, ?]>) // expected-error {{expected `strides` keyword after offset specification}}

// -----

func.func @memref_stride_missing_colon_2(memref<42x42xi8, offset: 0, strides [?, ?]>) // expected-error {{expected colon after `strides` keyword}}

// -----

// expected-error @+1 {{expected '['}}
func.func @memref_stride_invalid_strides(memref<42x42xi8, offset: 0, strides: ()>)

// -----

func.func @memref_zero_stride(memref<42x42xi8, offset: ?, strides: [0, ?]>) // expected-error {{invalid memref stride}}

// -----

func.func @tensor_encoding_mismatch(%arg0: tensor<8xi32, "enc">) -> (tensor<8xi32>) { // expected-note {{prior use here}}
  return %arg0: tensor<8xi32> // expected-error {{use of value '%arg0' expects different type than prior uses: 'tensor<8xi32>' vs 'tensor<8xi32, "enc">'}}
}

// -----

func.func @bad_branch() {
^bb12:
  cf.br ^missing  // expected-error {{reference to an undefined block}}
}

// -----

func.func @block_redef() {
^bb42:
  return
^bb42:        // expected-error {{redefinition of block '^bb42'}}
  return
}

// -----

func.func @no_terminator() {   // expected-error {{empty block: expect at least a terminator}}
^bb40:
  return
^bb41:
^bb42:
  return
}

// -----

func.func @block_no_rparen() {
^bb42 (%bb42 : i32: // expected-error {{expected ')'}}
  return
}

// -----

func.func @block_arg_no_ssaid() {
^bb42 (i32): // expected-error {{expected SSA operand}}
  return
}

// -----

func.func @block_arg_no_type() {
^bb42 (%0): // expected-error {{expected ':' and type for SSA operand}}
  return
}

// -----

func.func @block_arg_no_close_paren() {
^bb42:
  cf.br ^bb2( // expected-error {{expected ':'}}
  return
}

// -----

func.func @block_first_has_predecessor() {
// expected-error@-1 {{entry block of region may not have predecessors}}
^bb42:
  cf.br ^bb43
^bb43:
  cf.br ^bb42
}

// -----

func.func @no_return() {
  %x = arith.constant 0 : i32
  %y = arith.constant 1 : i32  // expected-error {{block with no terminator}}
}

// -----

func.func @no_terminator() {
  cf.br ^bb1
^bb1:
  %x = arith.constant 0 : i32
  %y = arith.constant 1 : i32  // expected-error {{block with no terminator}}
}

// -----

func.func @no_block_arg_enclosing_parens() {
^bb %x: i32 : // expected-error {{expected ':' after block name}}
  return
}

// -----

"       // expected-error {{expected}}
"

// -----

"       // expected-error {{expected}}

// -----

func.func @bad_op_type() {
^bb40:
  "foo"() : i32  // expected-error {{expected function type}}
  return
}
// -----

func.func @no_terminator() {
^bb40:
  "foo"() : ()->()
  ""() : ()->()  // expected-error {{empty operation name is invalid}}
  return
}

// -----

func.func @illegaltype(i21312312323120) // expected-error {{invalid integer width}}

// -----

func.func @malformed_for_percent() {
  affine.for i = 1 to 10 { // expected-error {{expected SSA operand}}

// -----

func.func @malformed_for_equal() {
  affine.for %i 1 to 10 { // expected-error {{expected '='}}

// -----

func.func @malformed_for_to() {
  affine.for %i = 1 too 10 { // expected-error {{expected 'to' between bounds}}
  }
}

// -----

func.func @incomplete_for() {
  affine.for %i = 1 to 10 step 2
}        // expected-error @-1 {{expected '{' to begin a region}}

// -----

#map0 = affine_map<(d0) -> (d0 floordiv 4)>

func.func @reference_to_iv_in_bound() {
  // expected-error@+2 {{region entry argument '%i0' is already in use}}
  // expected-note@+1 {{previously referenced here}}
  affine.for %i0 = #map0(%i0) to 10 {
  }
}

// -----

func.func @nonconstant_step(%1 : i32) {
  affine.for %2 = 1 to 5 step %1 { // expected-error {{expected attribute value}}

// -----

func.func @for_negative_stride() {
  affine.for %i = 1 to 10 step -1
}        // expected-error@-1 {{expected step to be representable as a positive signed integer}}

// -----

func.func @non_operation() {
  test.asd   // expected-error {{custom op 'test.asd' is unknown}}
}

// -----

func.func @non_operation() {
  // expected-error@+1 {{custom op 'asd' is unknown (tried 'func.asd' as well)}}
  asd
}

// -----

func.func @invalid_if_conditional2() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i >= )>  // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

func.func @invalid_if_conditional3() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i == 1)> // expected-error {{expected '0' after '=='}}
  }
}

// -----

func.func @invalid_if_conditional4() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i >= 2)> // expected-error {{expected '0' after '>='}}
  }
}

// -----

func.func @invalid_if_conditional5() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i)[N] : (i <= 0)> // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

func.func @invalid_if_conditional6() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i) : (i)> // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----
// TODO: support affine.if (1)?
func.func @invalid_if_conditional7() {
  affine.for %i = 1 to 10 {
    affine.if affine_set<(i) : (1)> // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

#map = affine_map<(d0) -> (%  // expected-error {{invalid SSA name}}

// -----

func.func @test() {
^bb40:
  %1 = "foo"() : (i32)->i64 // expected-error {{expected 0 operand types but had 1}}
  return
}

// -----

func.func @redef() {
^bb42:
  %x = "xxx"(){index = 0} : ()->i32 // expected-note {{previously defined here}}
  %x = "xxx"(){index = 0} : ()->i32 // expected-error {{redefinition of SSA value '%x'}}
  return
}

// -----

func.func @undef() {
^bb42:
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value}}
  return
}

// -----

func.func @malformed_type(%a : intt) { // expected-error {{expected non-function type}}
}

// -----

func.func @resulterror() -> i32 {
^bb42:
  return    // expected-error {{'func.return' op has 0 operands, but enclosing function (@resulterror) returns 1}}
}

// -----

func.func @func_resulterror() -> i32 {
  return // expected-error {{'func.return' op has 0 operands, but enclosing function (@func_resulterror) returns 1}}
}

// -----

func.func @argError() {
^bb1(%a: i64):  // expected-note {{previously defined here}}
  cf.br ^bb2
^bb2(%a: i64):  // expected-error{{redefinition of SSA value '%a'}}
  return
}

// -----

func.func @br_mismatch() {
^bb0:
  %0:2 = "foo"() : () -> (i1, i17)
  // expected-error @+1 {{branch has 2 operands for successor #0, but target block has 1}}
  cf.br ^bb1(%0#1, %0#0 : i17, i1)

^bb1(%x: i17):
  return
}

// -----

func.func @succ_arg_type_mismatch() {
^bb0:
  %0 = "getBool"() : () -> i1
  // expected-error @+1 {{type mismatch for bb argument #0 of successor #0}}
  cf.br ^bb1(%0 : i1)

^bb1(%x: i32):
  return
}


// -----

// Test no nested vector.
func.func @vectors(vector<1 x vector<1xi32>>, vector<2x4xf32>)
// expected-error@-1 {{vector elements must be int/index/float type}}

// -----

func.func @condbr_notbool() {
^bb0:
  %a = "foo"() : () -> i32 // expected-note {{prior use here}}
  cf.cond_br %a, ^bb0, ^bb0 // expected-error {{use of value '%a' expects different type than prior uses: 'i1' vs 'i32'}}
}

// -----

func.func @condbr_badtype() {
^bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cf.cond_br %c, ^bb0(%a, %a : i32, ^bb0) // expected-error {{expected non-function type}}
}

// -----

func.func @condbr_a_bb_is_not_a_type() {
^bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cf.cond_br %c, ^bb0(%a, %a : i32, i32), i32 // expected-error {{expected block name}}
}

// -----

func.func @successors_in_non_terminator(%a : i32, %b : i32) {
  %c = "arith.addi"(%a, %b)[^bb1] : () -> () // expected-error {{successors in non-terminator}}
^bb1:
  return
}

// -----

func.func @undef() {
^bb0:
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value name}}
  return
}

// -----

func.func @undef() {
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value name}}
  return
}

// -----

func.func @duplicate_induction_var() {
  affine.for %i = 1 to 10 {   // expected-note {{previously referenced here}}
    affine.for %i = 1 to 10 { // expected-error {{region entry argument '%i' is already in use}}
    }
  }
  return
}

// -----

func.func @name_scope_failure() {
  affine.for %i = 1 to 10 {
  }
  "xxx"(%i) : (index)->()   // expected-error {{use of undeclared SSA value name}}
  return
}

// -----

func.func @dominance_failure() {
^bb0:
  "foo"(%x) : (i32) -> ()    // expected-error {{operand #0 does not dominate this use}}
  cf.br ^bb1
^bb1:
  %x = "bar"() : () -> i32    // expected-note {{operand defined here (op in the same region)}}
  return
}

// -----

func.func @dominance_failure() {
^bb0:
  "foo"(%x) : (i32) -> ()    // expected-error {{operand #0 does not dominate this use}}
  %x = "bar"() : () -> i32    // expected-note {{operand defined here (op in the same block)}}
  cf.br ^bb1
^bb1:
  return
}

// -----

func.func @dominance_failure() {
  "foo"() ({
    "foo"(%x) : (i32) -> ()    // expected-error {{operand #0 does not dominate this use}}
  }) : () -> ()
  %x = "bar"() : () -> i32    // expected-note {{operand defined here (op in a parent region)}}
  return
}

// -----

func.func @dominance_failure() {  //  expected-note {{operand defined as a block argument (block #1 in the same region)}}
^bb0:
  cf.br ^bb1(%x : i32)    // expected-error {{operand #0 does not dominate this use}}
^bb1(%x : i32):
  return
}

// -----

func.func @dominance_failure() {  //  expected-note {{operand defined as a block argument (block #1 in a parent region)}}
^bb0:
  %f = "foo"() ({
    "foo"(%x) : (i32) -> ()    // expected-error {{operand #0 does not dominate this use}}
  }) : () -> (i32)
  cf.br ^bb1(%f : i32)
^bb1(%x : i32):
  return
}

// -----

func.func @return_type_mismatch() -> i32 {
  %0 = "foo"() : ()->f32
  return %0 : f32  // expected-error {{type of return operand 0 ('f32') doesn't match function result type ('i32') in function @return_type_mismatch}}
}

// -----

func.func @return_inside_loop() {
  affine.for %i = 1 to 100 {
    // expected-error@+1 {{'func.return' op expects parent op 'func.func'}}
    func.return
  }
  return
}

// -----

// expected-error@+1 {{expected three consecutive dots for an ellipsis}}
func.func @malformed_ellipsis_one(.)

// -----

// expected-error@+1 {{expected three consecutive dots for an ellipsis}}
func.func @malformed_ellipsis_two(..)

// -----

// expected-error@+1 {{expected non-function type}}
func.func @func_variadic(...)

// -----

func.func private @redef()  // expected-note {{see existing symbol definition here}}
func.func private @redef()  // expected-error {{redefinition of symbol named 'redef'}}

// -----

func.func @foo() {
^bb0:
  %x = constant @foo : (i32) -> ()  // expected-error {{reference to function with mismatched type}}
  return
}

// -----

func.func @undefined_function() {
^bb0:
  %x = constant @qux : (i32) -> ()  // expected-error {{reference to undefined function 'qux'}}
  return
}

// -----

#map1 = affine_map<(i)[j] -> (i+j)>

func.func @bound_symbol_mismatch(%N : index) {
  affine.for %i = #map1(%N) to 100 {
  // expected-error@-1 {{symbol operand count and affine map symbol count must match}}
  }
  return
}

// -----

#map1 = affine_map<(i)[j] -> (i+j)>

func.func @bound_dim_mismatch(%N : index) {
  affine.for %i = #map1(%N, %N)[%N] to 100 {
  // expected-error@-1 {{dim operand count and affine map dim count must match}}
  }
  return
}

// -----

func.func @large_bound() {
  affine.for %i = 1 to 9223372036854775810 {
  // expected-error@-1 {{integer constant out of range for attribute}}
  }
  return
}

// -----

func.func @max_in_upper_bound(%N : index) {
  affine.for %i = 1 to max affine_map<(i)->(N, 100)> { //expected-error {{expected attribute value}}
  }
  return
}

// -----

func.func @step_typo() {
  affine.for %i = 1 to 100 step -- 1 { //expected-error {{expected constant integer}}
  }
  return
}

// -----

func.func @invalid_bound_map(%N : i32) {
  affine.for %i = 1 to affine_map<(i)->(j)>(%N) { //expected-error {{use of undeclared identifier}}
  }
  return
}

// -----

// expected-error @+1 {{expected '(' in integer set constraint list}}
#set0 = affine_set<(i)[N, M] : )i >= 0)>

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands1(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0(%i) {
    // expected-error@-1 {{symbol operand count and integer set symbol count must match}}

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands2(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0()[%N] {
    // expected-error@-1 {{dim operand count and integer set dim count must match}}

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands3(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0(%i)[%i] {
    // expected-error@-1 {{operand cannot be used as a symbol}}
    }
  }
  return
}

// -----
// expected-error@+1 {{expected '"' in string literal}}
"J// -----
func.func @calls(%arg0: i32) {
  // expected-error@+1 {{expected non-function type}}
  %z = "casdasda"(%x) : (ppop32) -> i32
}
// -----
// expected-error@+1 {{expected SSA operand}}
func.func @n(){^b(
// -----

func.func @elementsattr_non_tensor_type() -> () {
^bb0:
  "foo"(){bar = dense<[4]> : i32} : () -> () // expected-error {{elements literal must be a ranked tensor or vector type}}
}

// -----

func.func @elementsattr_non_ranked() -> () {
^bb0:
  "foo"(){bar = dense<[4]> : tensor<?xi32>} : () -> () // expected-error {{elements literal type must have static shape}}
}

// -----

func.func @elementsattr_shape_mismatch() -> () {
^bb0:
  "foo"(){bar = dense<[4]> : tensor<5xi32>} : () -> () // expected-error {{inferred shape of elements literal ([1]) does not match type ([5])}}
}

// -----

func.func @elementsattr_invalid() -> () {
^bb0:
  "foo"(){bar = dense<[4, [5]]> : tensor<2xi32>} : () -> () // expected-error {{tensor literal is invalid; ranks are not consistent between elements}}
}

// -----

func.func @elementsattr_badtoken() -> () {
^bb0:
  "foo"(){bar = dense<[tf_opaque]> : tensor<1xi32>} : () -> () // expected-error {{expected element literal of primitive type}}
}

// -----

func.func @elementsattr_floattype1() -> () {
^bb0:
  // expected-error@+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = dense<[4.0]> : tensor<1xi32>} : () -> ()
}

// -----

func.func @elementsattr_floattype1() -> () {
^bb0:
  // expected-error@+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = dense<4.0> : tensor<i32>} : () -> ()
}

// -----

func.func @elementsattr_floattype2() -> () {
^bb0:
  // expected-error@+1 {{expected floating-point elements, but parsed integer}}
  "foo"(){bar = dense<[4]> : tensor<1xf32>} : () -> ()
}

// -----

func.func @elementsattr_toolarge1() -> () {
^bb0:
  "foo"(){bar = dense<[777]> : tensor<1xi8>} : () -> () // expected-error {{integer constant out of range}}
}

// -----

// expected-error@+1 {{parsed zero elements, but type ('tensor<i64>') expected at least 1}}
#attr = dense<> : tensor<i64>

// -----

func.func @elementsattr_toolarge2() -> () {
^bb0:
  "foo"(){bar = dense<[-777]> : tensor<1xi8>} : () -> () // expected-error {{integer constant out of range}}
}

// -----

"foo"(){bar = dense<[()]> : tensor<complex<i64>>} : () -> () // expected-error {{expected element literal of primitive type}}

// -----

"foo"(){bar = dense<[(10)]> : tensor<complex<i64>>} : () -> () // expected-error {{expected ',' between complex elements}}

// -----

"foo"(){bar = dense<[(10,)]> : tensor<complex<i64>>} : () -> () // expected-error {{expected element literal of primitive type}}

// -----

"foo"(){bar = dense<[(10,10]> : tensor<complex<i64>>} : () -> () // expected-error {{expected ')' after complex elements}}

// -----

func.func @elementsattr_malformed_opaque() -> () {
^bb0:
  "foo"(){bar = opaque<10, "0xQZz123"> : tensor<1xi8>} : () -> () // expected-error {{expected dialect namespace}}
}

// -----

func.func @elementsattr_malformed_opaque1() -> () {
^bb0:
  "foo"(){bar = opaque<"_", "0xQZz123"> : tensor<1xi8>} : () -> () // expected-error {{expected string containing hex digits starting with `0x`}}
}

// -----

func.func @elementsattr_malformed_opaque2() -> () {
^bb0:
  "foo"(){bar = opaque<"_", "00abc"> : tensor<1xi8>} : () -> () // expected-error {{expected string containing hex digits starting with `0x`}}
}

// -----

func.func @redundant_signature(%a : i32) -> () {
^bb0(%b : i32):  // expected-error {{invalid block name in region with named arguments}}
  return
}

// -----

func.func @mixed_named_arguments(%a : i32,
                               f32) -> () {
    // expected-error @-1 {{expected SSA identifier}}
  return
}

// -----

func.func @mixed_named_arguments(f32,
                               %a : i32) -> () { // expected-error {{expected type instead of SSA identifier}}
  return
}

// -----

// This used to crash the parser, but should just error out by interpreting
// `tensor` as operator rather than as a type.
func.func @f(f32) {
^bb0(%a : f32):
  %18 = arith.cmpi slt, %idx, %idx : index
  tensor<42 x index  // expected-error {{custom op 'tensor' is unknown (tried 'func.tensor' as well)}}
  return
}

// -----

func.func @f(%m : memref<?x?xf32>) {
  affine.for %i0 = 0 to 42 {
    // expected-note@+1 {{previously referenced here}}
    %x = memref.load %m[%i0, %i1] : memref<?x?xf32>
  }
  // expected-error@+1 {{region entry argument '%i1' is already in use}}
  affine.for %i1 = 0 to 42 {
  }
  return
}

// -----

func.func @dialect_type_empty_namespace(!<"">) -> () { // expected-error {{invalid type identifier}}
  return
}

// -----

func.func @dialect_type_no_string_type_data(!foo<>) -> () { // expected-error {{expected string literal data in dialect symbol}}
  return
}

// -----

func.func @dialect_type_missing_greater(!foo<"") -> () { // expected-error {{expected '>' in dialect symbol}}
  return
}

// -----

func.func @type_alias_unknown(!unknown_alias) -> () { // expected-error {{undefined symbol alias id 'unknown_alias'}}
  return
}

// -----

// expected-error @+1 {{type names with a '.' are reserved for dialect-defined names}}
!foo.bar = i32

// -----

!missing_eq_alias i32 // expected-error {{expected '=' in type alias definition}}

// -----

!missing_type_alias = // expected-error {{expected non-function type}}

// -----

!redef_alias = i32
!redef_alias = i32 // expected-error {{redefinition of type alias id 'redef_alias'}}

// -----

// Check ill-formed opaque tensor.
func.func @complex_loops() {
  affine.for %i1 = 1 to 100 {
  // expected-error @+1 {{expected '"' in string literal}}
  "opaqueIntTensor"(){bar = opaque<"_", "0x686]> : tensor<2x1x4xi32>} : () -> ()

// -----

func.func @mi() {
  // expected-error @+1 {{expected element literal of primitive type}}
  "fooi64"(){bar = sparse<vector<1xi64>,[,[,1]

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{expected 1-d tensor for sparse element values}}
  "foof16"(){bar = sparse<[[0, 0, 0]],  [[-2.0]]> : vector<1x1x1xf16>} : () -> ()

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{expected element literal of primitive type}}
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0,, [[0, 0, 0]], [-2.0]> : tensor<2x2x2xi16>} : () -> ()

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{sparse index #0 is not contained within the value shape, with index=[1, 1], and type='tensor<1x1xi16>'}}
  "fooi16"(){bar = sparse<1, 10> : tensor<1x1xi16>} : () -> ()

// -----

func.func @invalid_affine_structure() {
  %c0 = arith.constant 0 : index
  %idx = affine.apply affine_map<(d0, d1)> (%c0, %c0) // expected-error {{expected '->' or ':'}}
  return
}

// -----

func.func @missing_for_max(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // expected-error @+1 {{lower loop bound affine map with multiple results requires 'max' prefix}}
  affine.for %i0 = affine_map<()[s]->(0,s-1)>()[%arg0] to %arg1 {
  }
  return
}

// -----

func.func @missing_for_min(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // expected-error @+1 {{upper loop bound affine map with multiple results requires 'min' prefix}}
  affine.for %i0 = %arg0 to affine_map<()[s]->(100,s+1)>()[%arg1] {
  }
  return
}

// -----

// expected-error @+1 {{vector types must have positive constant sizes}}
func.func @zero_vector_type() -> vector<0xi32>

// -----

// expected-error @+1 {{vector types must have positive constant sizes}}
func.func @zero_in_vector_type() -> vector<1x0xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_vector_size() -> vector<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_vector_size() -> vector<1x-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_memref_size() -> memref<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_memref_size() -> memref<1x-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_tensor_size() -> tensor<-1xi32>

// -----

// expected-error @+1 {{expected non-function type}}
func.func @negative_in_tensor_size() -> tensor<1x-1xi32>

// -----

func.func @invalid_nested_dominance() {
  "test.ssacfg_region"() ({
    // expected-error @+1 {{operand #0 does not dominate this use}}
    "foo.use" (%1) : (i32) -> ()
    cf.br ^bb2

  ^bb2:
    // expected-note @+1 {{operand defined here}}
    %1 = arith.constant 0 : i32
    "foo.yield" () : () -> ()
  }) : () -> ()
  return
}

// -----

// expected-error @+1 {{unbalanced ']' character in pretty dialect name}}
func.func @invalid_unknown_type_dialect_name() -> !invalid.dialect<!x@#]!@#>

// -----

// expected-error @+1 {{@ identifier expected to start with letter or '_'}}
func.func @$invalid_function_name()

// -----

// expected-error @+1 {{arguments may only have dialect attributes}}
func.func @invalid_func_arg_attr(i1 {non_dialect_attr = 10})

// -----

// expected-error @+1 {{results may only have dialect attributes}}
func.func @invalid_func_result_attr() -> (i1 {non_dialect_attr = 10})

// -----

// expected-error @+1 {{expected '<' in tuple type}}
func.func @invalid_tuple_missing_less(tuple i32>)

// -----

// expected-error @+1 {{expected '>' in tuple type}}
func.func @invalid_tuple_missing_greater(tuple<i32)

// -----

// Should not crash because of deletion order here.
func.func @invalid_region_dominance() {
  "foo.use" (%1) : (i32) -> ()
  "foo.region"() ({
    %1 = arith.constant 0 : i32  // This value is used outside of the region.
    "foo.yield" () : () -> ()
  }, {
    // expected-error @+1 {{expected operation name in quotes}}
    %2 = arith.constant 1 i32  // Syntax error causes region deletion.
  }) : () -> ()
  return
}

// -----

// Should not crash because of deletion order here.
func.func @invalid_region_block() {
  "foo.branch"()[^bb2] : () -> ()  // Attempt to jump into the region.

^bb1:
  "foo.region"() ({
    ^bb2:
      "foo.yield"() : () -> ()
  }, {
    // expected-error @+1 {{expected operation name in quotes}}
    %2 = arith.constant 1 i32  // Syntax error causes region deletion.
  }) : () -> ()
}

// -----

// Should not crash because of deletion order here.
func.func @invalid_region_dominance() {
  "foo.use" (%1) : (i32) -> ()
  "foo.region"() ({
    "foo.region"() ({
      %1 = arith.constant 0 : i32  // This value is used outside of the region.
      "foo.yield" () : () -> ()
    }) : () -> ()
  }, {
    // expected-error @+1 {{expected operation name in quotes}}
    %2 = arith.constant 1 i32  // Syntax error causes region deletion.
  }) : () -> ()
  return
}

// -----

func.func @unfinished_region_list() {
  // expected-error@+1 {{expected ')' to end region list}}
  "region"() ({},{},{} : () -> ()
}

// -----

func.func @multi_result_missing_count() {
  // expected-error@+1 {{expected integer number of results}}
  %0: = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @multi_result_zero_count() {
  // expected-error@+1 {{expected named operation to have at least 1 result}}
  %0:0 = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @multi_result_invalid_identifier() {
  // expected-error@+1 {{expected valid ssa identifier}}
  %0, = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @multi_result_mismatch_count() {
  // expected-error@+1 {{operation defines 2 results but was provided 1 to bind}}
  %0:1 = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @multi_result_mismatch_count() {
  // expected-error@+1 {{operation defines 2 results but was provided 3 to bind}}
  %0, %1, %3 = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @no_result_with_name() {
  // expected-error@+1 {{cannot name an operation with no results}}
  %0 = "foo" () : () -> ()
  return
}

// -----

func.func @conflicting_names() {
  // expected-note@+1 {{previously defined here}}
  %foo, %bar  = "foo" () : () -> (i32, i32)

  // expected-error@+1 {{redefinition of SSA value '%bar'}}
  %bar, %baz  = "foo" () : () -> (i32, i32)
  return
}

// -----

func.func @ssa_name_missing_eq() {
  // expected-error@+1 {{expected '=' after SSA name}}
  %0:2 "foo" () : () -> (i32, i32)
  return
}

// -----

// expected-error @+1 {{invalid element type for complex}}
func.func @bad_complex(complex<memref<2x4xi8>>)

// -----

// expected-error @+1 {{expected '<' in complex type}}
func.func @bad_complex(complex memref<2x4xi8>>)

// -----

// expected-error @+1 {{expected '>' in complex type}}
func.func @bad_complex(complex<i32)

// -----

// expected-error @+1 {{attribute names with a '.' are reserved for dialect-defined names}}
#foo.attr = i32

// -----

func.func @invalid_region_dominance() {
  "test.ssacfg_region"() ({
    // expected-error @+1 {{operand #0 does not dominate this use}}
    "foo.use" (%def) : (i32) -> ()
    "foo.yield" () : () -> ()
  }, {
    // expected-note @+1 {{operand defined here}}
    %def = "foo.def" () : () -> i32
  }) : () -> ()
  return
}

// -----

func.func @invalid_region_dominance() {
  // expected-note @+1 {{operand defined here}}
  %def = "test.ssacfg_region"() ({
    // expected-error @+1 {{operand #0 does not dominate this use}}
    "foo.use" (%def) : (i32) -> ()
    "foo.yield" () : () -> ()
  }) : () -> (i32)
  return
}

// -----

func.func @hexadecimal_float_leading_minus() {
  // expected-error @+1 {{hexadecimal float literal should not have a leading minus}}
  "foo"() {value = -0x7fff : f16} : () -> ()
}

// -----

func.func @hexadecimal_float_literal_overflow() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {value = 0xffffffff : f16} : () -> ()
}

// -----

func.func @decimal_float_literal() {
  // expected-error @+2 {{unexpected decimal integer literal for a floating point value}}
  // expected-note @+1 {{add a trailing dot to make the literal a float}}
  "foo"() {value = 42 : f32} : () -> ()
}

// -----

func.func @float_in_int_tensor() {
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"() {bar = dense<[42.0, 42]> : tensor<2xi32>} : () -> ()
}

// -----

func.func @float_in_bool_tensor() {
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"() {bar = dense<[true, 42.0]> : tensor<2xi1>} : () -> ()
}

// -----

func.func @decimal_int_in_float_tensor() {
  // expected-error @+1 {{expected floating-point elements, but parsed integer}}
  "foo"() {bar = dense<[42, 42.0]> : tensor<2xf32>} : () -> ()
}

// -----

func.func @bool_in_float_tensor() {
  // expected-error @+1 {{expected floating-point elements, but parsed integer}}
  "foo"() {bar = dense<[42.0, true]> : tensor<2xf32>} : () -> ()
}

// -----

func.func @hexadecimal_float_leading_minus_in_tensor() {
  // expected-error @+1 {{hexadecimal float literal should not have a leading minus}}
  "foo"() {bar = dense<-0x7FFFFFFF> : tensor<2xf32>} : () -> ()
}

// -----

// Check that we report an error when a value could be parsed, but does not fit
// into the specified type.
func.func @hexadecimal_float_too_wide_for_type_in_tensor() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {bar = dense<0x7FF0000000000000> : tensor<2xf32>} : () -> ()
}

// -----

// Check that we report an error when a value is too wide to be parsed.
func.func @hexadecimal_float_too_wide_in_tensor() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {bar = dense<0x7FFFFFF0000000000000> : tensor<2xf32>} : () -> ()
}

// -----

func.func @integer_too_wide_in_tensor() {
  // expected-error @+1 {{integer constant out of range for type}}
  "foo"() {bar = dense<0xFFFFFFFFFFFFFF> : tensor<2xi16>} : () -> ()
}

// -----

func.func @bool_literal_in_non_bool_tensor() {
  // expected-error @+1 {{expected i1 type for 'true' or 'false' values}}
  "foo"() {bar = dense<true> : tensor<2xi16>} : () -> ()
}

// -----

// expected-error @+1 {{unbalanced ')' character in pretty dialect name}}
func.func @bad_arrow(%arg : !unreg.ptr<(i32)->)

// -----

func.func @negative_value_in_unsigned_int_attr() {
  // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
  "foo"() {bar = -5 : ui32} : () -> ()
}

// -----

func.func @negative_value_in_unsigned_vector_attr() {
  // expected-error @+1 {{expected unsigned integer elements, but parsed negative value}}
  "foo"() {bar = dense<[5, -5]> : vector<2xui32>} : () -> ()
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -129 : i8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 256 : i8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -129 : si8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 129 : si8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 256 : ui8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -32769 : i16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 65536 : i16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -32769 : si16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 32768 : si16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 65536: ui16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -2147483649 : i32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 4294967296 : i32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -2147483649 : si32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 2147483648 : si32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 4294967296 : ui32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -9223372036854775809 : i64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 18446744073709551616 : i64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -9223372036854775809 : si64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 9223372036854775808 : si64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 18446744073709551616 : ui64
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 79228162514264337593543950336 : ui96
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 79228162514264337593543950336 : i96
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 39614081257132168796771975168 : si96
  } : () -> ()
  return
}

// -----

func.func @duplicate_dictionary_attr_key() {
  // expected-error @+1 {{duplicate key 'a' in dictionary attribute}}
  "foo.op"() {a, a} : () -> ()
}

// -----

// expected-error @+1 {{attribute 'attr' occurs more than once in the attribute list}}
test.format_symbol_name_attr_op @name { attr = "xx" }

// -----

func.func @forward_reference_type_check() -> (i8) {
  cf.br ^bb2

^bb1:
  // expected-note @+1 {{previously used here with type 'i8'}}
  return %1 : i8

^bb2:
  // expected-error @+1 {{definition of SSA value '%1#0' has type 'f32'}}
  %1 = "bar"() : () -> (f32)
  cf.br ^bb1
}

// -----

func.func @dominance_error_in_unreachable_op() -> i1 {
  %c = arith.constant false
  return %c : i1
^bb0:
  "test.ssacfg_region" () ({ // unreachable
    ^bb1:
// expected-error @+1 {{operand #0 does not dominate this use}}
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
      cf.br ^bb4
    ^bb2:
      cf.br ^bb2
    ^bb4:
      %1 = "foo"() : ()->i64   // expected-note {{operand defined here}}
  }) : () -> ()
  return %c : i1
}

// -----

func.func @invalid_region_dominance_with_dominance_free_regions() {
  test.graph_region {
    "foo.use" (%1) : (i32) -> ()
    "foo.region"() ({
      %1 = arith.constant 0 : i32  // This value is used outside of the region.
      "foo.yield" () : () -> ()
    }, {
      // expected-error @+1 {{expected operation name in quotes}}
      %2 = arith.constant 1 i32  // Syntax error causes region deletion.
    }) : () -> ()
  }
  return
}

// -----

func.func @foo() {} // expected-error {{expected non-empty function body}}

// -----

// expected-error@+1 {{expected valid attribute name}}
"t"(){""}

// -----

// expected-error@+1 {{expected ']'}}
"f"() { b = [@m:

// -----

// This makes sure we emit an error at the end of the correct line, the : is
// expected at the end of foo, not on the return line.
func.func @error_at_end_of_line() {
  // expected-error@+1 {{expected ':' followed by operation type}}
  %0 = "foo"() 
  return
}

// -----

// This makes sure we emit an error at the end of the correct line, the : is
// expected at the end of foo, not on the return line.
func.func @error_at_end_of_line() {
  %0 = "foo"() 
  // expected-error@-1 {{expected ':' followed by operation type}}

  // This is a comment and so is the thing above.
  return
}

// -----

// This makes sure we emit an error at the end of the correct line, the : is
// expected at the end of foo, not on the return line.
// This shows that it backs up to before the comment.
func.func @error_at_end_of_line() {
  %0 = "foo"()  // expected-error {{expected ':' followed by operation type}}
  return
}

// -----

@foo   // expected-error {{expected operation name in quotes}}

// -----

func.func @func() {
  %c0 = arith.constant  // expected-error {{expected attribute value}}

  %x = arith.constant 1 : i32
}

