// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// Check different error cases.

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

func.func @non_operation() {
  test.asd   // expected-error {{custom op 'test.asd' is unknown}}
}

// -----

func.func @non_operation() {
  // expected-error@+1 {{custom op 'asd' is unknown (tried 'func.asd' as well)}}
  asd
}

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

// expected-error@+1 {{expected three consecutive dots for an ellipsis}}
func.func @malformed_ellipsis_one(.)

// -----

// expected-error@+1 {{expected three consecutive dots for an ellipsis}}
func.func @malformed_ellipsis_two(..)

// -----

func.func private @redef()  // expected-note {{see existing symbol definition here}}
func.func private @redef()  // expected-error {{redefinition of symbol named 'redef'}}

// -----

func.func @calls(%arg0: i32) {
  // expected-error@+1 {{expected non-function type}}
  %z = "casdasda"(%x) : (ppop32) -> i32
}

// -----

// expected-error@+1 {{expected SSA operand}}
func.func @n(){^b(

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

func.func @dialect_type_missing_greater(!foo<) -> () { // expected-error {{unbalanced ')' character in pretty dialect name}}
  return

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

// expected-error @+1 {{unbalanced ')' character in pretty dialect name}}
func.func @bad_arrow(%arg : !unreg.ptr<(i32)->)

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

// expected-error@+1 {{expected valid attribute name}}
"t"(){""}

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
