'use strict';

const builtin_dialect = require('./dialect/builtin');
const func_dialect = require('./dialect/func');
const llvm_dialect = require('./dialect/llvm');
const arith_dialect = require('./dialect/arith');
const math_dialect = require('./dialect/math');
const cf_dialect = require('./dialect/cf');
const scf_dialect = require('./dialect/scf');
const memref_dialect = require('./dialect/memref');
const vector_dialect = require('./dialect/vector');
const tensor_dialect = require('./dialect/tensor');
const bufferization_dialect = require('./dialect/bufferization');
const affine_dialect = require('./dialect/affine');
const linalg_dialect = require('./dialect/linalg');

const common = {
  // Top level production:
  //   (operation | attribute-alias-def | type-alias-def)
  toplevel : $ => seq($._toplevel, repeat($._toplevel)),
  _toplevel : $ => choice($.operation, $.attribute_alias_def, $.type_alias_def),

  // Common syntax (lang-ref)
  //  digit     ::= [0-9]
  //  hex_digit ::= [0-9a-fA-F]
  //  letter    ::= [a-zA-Z]
  //  id-punct  ::= [$._-]
  //
  //  integer-literal ::= decimal-literal | hexadecimal-literal
  //  decimal-literal ::= digit+
  //  hexadecimal-literal ::= `0x` hex_digit+
  //  float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
  //  string-literal  ::= `"` [^"\n\f\v\r]* `"`   TODO: define escaping rules
  //
  _digit : $ => /[0-9]/,
  integer_literal : $ => choice($._decimal_literal, $._hexadecimal_literal),
  _decimal_literal : $ => token(seq(optional(/[-+]/), repeat1(/[0-9]/))),
  _hexadecimal_literal : $ => token(seq('0x', repeat1(/[0-9a-fA-F]/))),
  float_literal : $ =>
      token(seq(optional(/[-+]/), repeat1(/[0-9]/), '.', repeat(/[0-9]/),
                optional(seq(/[eE]/, optional(/[-+]/), repeat1(/[0-9]/))))),
  string_literal : $ => token(seq('"', repeat(/[^\\"\n\f\v\r]+/), '"')),
  bool_literal : $ => token(choice('true', 'false')),
  unit_literal : $ => token('unit'),
  complex_literal : $ =>
      seq('(', choice($.integer_literal, $.float_literal), ',',
          choice($.integer_literal, $.float_literal), ')'),
  tensor_literal : $ =>
      seq(token(choice('dense', 'sparse')), '<',
          optional(choice(
              seq($.nested_idx_list, repeat(seq(',', $.nested_idx_list))),
              $._primitive_idx_literal)),
          '>'),
  array_literal : $ => seq(token('array'), '<', $.type, ':', $._idx_list, '>'),
  _literal : $ => choice($.integer_literal, $.float_literal, $.string_literal,
                         $.bool_literal, $.tensor_literal, $.array_literal,
                         $.complex_literal, $.unit_literal),

  nested_idx_list : $ =>
      seq('[', optional(choice($.nested_idx_list, $._idx_list)),
          repeat(seq(',', $.nested_idx_list)), ']'),
  _idx_list : $ => prec.right(seq($._primitive_idx_literal,
                                  repeat(seq(',', $._primitive_idx_literal)))),
  _primitive_idx_literal : $ => choice($.integer_literal, $.float_literal,
                                       $.bool_literal, $.complex_literal),

  // Identifiers
  //   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
  //   bare-id-list ::= bare-id (`,` bare-id)*
  //   value-id ::= `%` suffix-id
  //   suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
  //   alias-name :: = bare-id
  //
  //   symbol-ref-id ::= `@` (suffix-id | string-literal) (`::`
  //                     symbol-ref-id)?
  //   value-id-list ::= value-id (`,` value-id)*
  //
  //   // Uses of value, e.g. in an operand list to an operation.
  //   value-use ::= value-id
  //   value-use-list ::= value-use (`,` value-use)*
  bare_id : $ => token(seq(/[a-zA-Z_]/, repeat(/[a-zA-Z0-9_$.]/))),
  _alias_or_dialect_id : $ => token(seq(/[a-zA-Z_]/, repeat(/[a-zA-Z0-9_$]/))),
  bare_id_list : $ => seq($.bare_id, repeat(seq(',', $.bare_id))),
  value_use : $ => seq('%', $._suffix_id),
  _suffix_id : $ => token(seq(
      choice(repeat1(/[0-9]/), seq(/[a-zA-Z_$.-]/, repeat(/[a-zA-Z0-9_$.-]/))),
      optional(seq(choice(':', '#'), repeat1(/[0-9]/))))),
  symbol_ref_id : $ => seq('@', choice($._suffix_id, $.string_literal),
                           optional(seq('::', $.symbol_ref_id))),
  _value_use_list : $ => seq($.value_use, repeat(seq(',', $.value_use))),

  // Operations
  //   operation            ::= op-result-list? (generic-operation |
  //                            custom-operation)
  //                            trailing-location?
  //   generic-operation    ::= string-literal `(` value-use-list? `)`
  //                            successor-list? region-list?
  //                            dictionary-attribute? `:` function-type
  //   custom-operation     ::= bare-id custom-operation-format
  //   op-result-list       ::= op-result (`,` op-result)* `=`
  //   op-result            ::= value-id (`:` integer-literal)
  //   successor-list       ::= `[` successor (`,` successor)* `]`
  //   successor            ::= caret-id (`:` bb-arg-list)?
  //   region-list          ::= `(` region (`,` region)* `)`
  //   dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)?
  //                            `}`
  //   trailing-location    ::= (`loc` `(` location `)`)?
  operation : $ =>
      seq(field('lhs', optional($._op_result_list)),
          field('rhs', choice($.generic_operation, $.custom_operation)),
          field('location', optional($.trailing_location))),
  generic_operation : $ => seq(
      $.string_literal, $._value_use_list_parens, optional($._successor_list),
      optional($._region_list), optional($.attribute), ':', $.function_type),
  // custom-operation rule is defined later in the grammar, post the generic.
  _op_result_list : $ => seq($.op_result, repeat(seq(',', $.op_result)), '='),
  op_result : $ => seq($.value_use, optional(seq(':', $.integer_literal))),
  _successor_list : $ =>
      seq('[', $.successor, repeat(seq(',', $.successor)), ']'),
  successor : $ => seq($.caret_id, optional($._value_arg_list)),
  _region_list : $ => seq('(', $.region, repeat(seq(',', $.region)), ')'),
  dictionary_attribute : $ => seq('{', optional($.attribute_entry),
                                  repeat(seq(',', $.attribute_entry)), '}'),
  trailing_location : $ => seq(token('loc'), '(', $.location, ')'),
  // TODO: Complete location forms.
  location : $ => $.string_literal,

  // Blocks
  //   block           ::= block-label operation+
  //   block-label     ::= block-id block-arg-list? `:`
  //   block-id        ::= caret-id
  //   caret-id        ::= `^` suffix-id
  //   value-id-and-type ::= value-id `:` type
  //
  //   // Non-empty list of names and types.
  //   value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*
  //
  //   block-arg-list ::= `(` value-id-and-type-list? `)`
  block : $ => seq($.block_label, repeat1($.operation)),
  block_label : $ => seq($._block_id, optional($.block_arg_list), ':'),
  _block_id : $ => $.caret_id,
  caret_id : $ => seq('^', $._suffix_id),
  _value_use_and_type : $ => seq($.value_use, optional(seq(':', $.type))),
  _value_use_and_type_list : $ =>
      seq($._value_use_and_type, repeat(seq(',', $._value_use_and_type))),
  block_arg_list : $ => seq('(', optional($._value_use_and_type_list), ')'),
  _value_arg_list : $ => seq('(', optional($._value_use_type_list), ')'),
  _value_use_type_list : $ => seq($._value_use_list, $._type_annotation),

  // Regions
  //   region      ::= `{` entry-block? block* `}`
  //   entry-block ::= operation+
  region : $ => seq('{', optional($.entry_block), repeat($.block), '}'),
  entry_block : $ => repeat1($.operation),

  // Types
  //   type ::= type-alias | dialect-type | builtin-type
  //
  //   type-list-no-parens ::=  type (`,` type)*
  //   type-list-parens ::= `(` type-list-no-parens? `)`
  //
  //   // This is a common way to refer to a value with a specified type.
  //   ssa-use-and-type ::= ssa-use `:` type
  //   ssa-use ::= value-use
  //
  //   // Non-empty list of names and types.
  //   ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
  //
  //   function-type ::= (type | type-list-parens) `->` (type |
  //                      type-list-parens)
  type : $ => choice($.type_alias, $.dialect_type, $.builtin_type),
  _type_list_no_parens : $ => prec.left(seq($.type, repeat(seq(',', $.type)))),
  _type_list_parens : $ => seq('(', optional($._type_list_no_parens), ')'),
  function_type : $ =>
      seq(choice($.type, $._type_list_parens), $._function_return),
  _function_return : $ => seq(token('->'), choice($.type, $._type_list_parens)),
  _type_annotation : $ =>
      seq(':', choice(seq($.type, choice('from', 'into', 'to'), $.type),
                      $._type_list_no_parens)),
  _function_type_annotation : $ => seq(':', $.function_type),
  _literal_and_type : $ => seq($._literal, optional($._type_annotation)),

  // Type aliases
  //   type-alias-def ::= '!' alias-name '=' type
  //   type-alias ::= '!' alias-name
  type_alias_def : $ => seq('!', $._alias_or_dialect_id, '=', $.type),
  type_alias : $ => seq('!', $._alias_or_dialect_id),

  // Dialect Types
  //   dialect-namespace ::= bare-id
  //
  //   opaque-dialect-item ::= dialect-namespace '<' string-literal '>'
  //
  //   pretty-dialect-item ::= dialect-namespace '.'
  //   pretty-dialect-item-lead-ident pretty-dialect-item-body?
  //
  //   pretty-dialect-item-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'
  //   pretty-dialect-item-body ::= '<' pretty-dialect-item-contents+ '>'
  //   pretty-dialect-item-contents ::= pretty-dialect-item-body
  //                                 | '(' pretty-dialect-item-contents+ ')'
  //                                 | '[' pretty-dialect-item-contents+ ']'
  //                                 | '{' pretty-dialect-item-contents+ '}'
  //                                 | '[^[<({>\])}\0]+'
  //
  //   dialect-type ::= '!' (opaque-dialect-item | pretty-dialect-item)
  dialect_type : $ =>
      seq('!', choice($.opaque_dialect_item, $.pretty_dialect_item)),
  dialect_namespace : $ => $._alias_or_dialect_id,
  dialect_ident : $ => $._alias_or_dialect_id,
  opaque_dialect_item : $ =>
      seq($.dialect_namespace, '<', $.string_literal, '>'),
  pretty_dialect_item : $ => seq($.dialect_namespace, '.', $.dialect_ident,
                                 optional($.pretty_dialect_item_body)),
  pretty_dialect_item_body : $ =>
      seq('<', repeat($._pretty_dialect_item_contents), '>'),
  _pretty_dialect_item_contents : $ =>
      prec.left(choice($.pretty_dialect_item_body, repeat1(/[^<>]/))),

  // Builtin types
  builtin_type : $ => choice(
      // TODO: Add opaque_type
      $.integer_type, $.float_type, $.complex_type, $.index_type, $.memref_type,
      $.none_type, $.tensor_type, $.vector_type, $.tuple_type),

  // signed-integer-type ::= `si`[1-9][0-9]*
  // unsigned-integer-type ::= `ui`[1-9][0-9]*
  // signless-integer-type ::= `i`[1-9][0-9]*
  // integer-type ::= signed-integer-type | unsigned-integer-type |
  // signless-integer-type
  integer_type : $ =>
      token(seq(choice('si', 'ui', 'i'), /[1-9]/, repeat(/[0-9]/))),
  float_type : $ => token(
      choice('f16', 'f32', 'f64', 'f80', 'f128', 'bf16', 'f8E3M4', 'f8E4M3FN',
             'f8E4M3', 'f8E5M2', 'f6E2M3FN', 'f6E3M2FN')),
  index_type : $ => token('index'),
  none_type : $ => token('none'),
  complex_type : $ => seq(token('complex'), '<', $._prim_type, '>'),
  _prim_type : $ =>
      choice($.integer_type, $.float_type, $.index_type, $.complex_type,
             $.none_type, $.memref_type, $.vector_type, $.tensor_type),

  // memref-type ::= `memref` `<` dimension-list-ranked type
  //                 (`,` layout-specification)? (`,` memory-space)? `>`
  // layout-specification ::= attribute-value
  // memory-space ::= attribute-value
  memref_type : $ =>
      seq(token('memref'), '<', field('dimension_list', $.dim_list),
          optional(seq(',', $.attribute_value)),
          optional(seq(',', $.attribute_value)), '>'),
  dim_list : $ => seq($._dim_primitive, repeat(seq('x', $._dim_primitive))),
  _dim_primitive : $ => choice($._prim_type, repeat1($._digit), '?', '*'),

  // tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
  // dimension-list ::= (dimension `x`)*
  // dimension ::= `?` | decimal-literal
  // encoding ::= attribute-value
  // tensor-type ::= `tensor` `<` `*` `x` type `>`
  tensor_type : $ => seq(token('tensor'), '<', $.dim_list,
                         optional(seq(',', $.tensor_encoding)), '>'),
  tensor_encoding : $ => $.attribute_value,

  // vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
  // vector-element-type ::= float-type | integer-type | index-type
  // vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
  // static-dim-list ::= decimal-literal (`x` decimal-literal)*
  vector_type : $ =>
      seq(token('vector'), '<', optional($.vector_dim_list), $._prim_type, '>'),
  vector_dim_list : $ =>
      choice(seq($._static_dim_list, 'x',
                 optional(seq('[', $._static_dim_list, ']', 'x'))),
             seq('[', $._static_dim_list, ']', 'x')),
  _static_dim_list : $ =>
      seq(repeat1($._digit), repeat(seq('x', repeat1($._digit)))),

  // tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
  tuple_type : $ =>
      seq(token('tuple'), '<', $.tuple_dim, repeat(seq(',', $.tuple_dim)), '>'),
  tuple_dim : $ => $._prim_type,

  // Attributes
  //   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
  //   attribute-value ::= attribute-alias | dialect-attribute |
  //   builtin-attribute
  attribute_entry : $ => seq(choice($.bare_id, $.string_literal),
                             optional(seq('=', $.attribute_value))),
  attribute_value : $ =>
      choice(seq('[', optional($._attribute_value_nobracket),
                 repeat(seq(',', $._attribute_value_nobracket)), ']'),
             $._attribute_value_nobracket),
  _attribute_value_nobracket : $ =>
      choice($.attribute_alias, $.dialect_attribute, $.builtin_attribute,
             $.dictionary_attribute, $._literal_and_type, $.type),
  attribute : $ => choice($.attribute_alias, $.dialect_attribute,
                          $.builtin_attribute, $.dictionary_attribute),

  // Attribute Value Aliases
  //   attribute-alias-def ::= '#' alias-name '=' attribute-value
  //   attribute-alias ::= '#' alias-name
  attribute_alias_def : $ =>
      seq('#', $._alias_or_dialect_id, '=', $.attribute_value),
  attribute_alias : $ => seq('#', $._alias_or_dialect_id),

  // Dialect Attribute Values
  dialect_attribute : $ =>
      seq('#', choice($.opaque_dialect_item, $.pretty_dialect_item)),

  // Builtin Attribute Values
  builtin_attribute : $ => choice(
      // TODO
      $.strided_layout, $.affine_map, $.affine_set),
  strided_layout : $ => seq(token('strided'), '<', '[', $._dim_list_comma, ']',
                            optional(seq(',', token('offset'), ':',
                                         choice($.integer_literal, '?', '*'))),
                            '>'),
  _dim_list_comma : $ =>
      seq($._dim_primitive, repeat(seq(',', $._dim_primitive))),

  affine_map : $ =>
      seq(token('affine_map'), '<', $._multi_dim_affine_expr_parens,
          optional($._multi_dim_affine_expr_sq), token('->'),
          $._multi_dim_affine_expr_parens, '>'),
  affine_set : $ =>
      seq(token('affine_set'), '<', $._multi_dim_affine_expr_parens,
          optional($._multi_dim_affine_expr_sq), ':',
          $._multi_dim_affine_expr_parens, '>'),
  _multi_dim_affine_expr_parens : $ =>
      seq('(', optional($._multi_dim_affine_expr), ')'),
  _multi_dim_affine_expr_sq : $ =>
      seq('[', optional($._multi_dim_affine_expr), ']'),

  // affine-expr ::= `(` affine-expr `)`
  //               | affine-expr `+` affine-expr
  //               | affine-expr `-` affine-expr
  //               | `-`? integer-literal `*` affine-expr
  //               | affine-expr `ceildiv` integer-literal
  //               | affine-expr `floordiv` integer-literal
  //               | affine-expr `mod` integer-literal
  //               | `-`affine-expr
  //               | bare-id
  //               | `-`? integer-literal
  // multi-dim-affine-expr ::= `(` `)`
  //                         | `(` affine-expr (`,` affine-expr)* `)`

  // semi-affine-expr ::= `(` semi-affine-expr `)`
  //                    | semi-affine-expr `+` semi-affine-expr
  //                    | semi-affine-expr `-` semi-affine-expr
  //                    | symbol-or-const `*` semi-affine-expr
  //                    | semi-affine-expr `ceildiv` symbol-or-const
  //                    | semi-affine-expr `floordiv` symbol-or-const
  //                    | semi-affine-expr `mod` symbol-or-const
  //                    | bare-id
  //                    | `-`? integer-literal
  // symbol-or-const ::= `-`? integer-literal | symbol-id
  // multi-dim-semi-affine-expr ::= `(` semi-affine-expr (`,` semi-affine-expr)*
  // `)`

  // affine-constraint ::= affine-expr `>=` `affine-expr`
  //                     | affine-expr `<=` `affine-expr`
  //                     | affine-expr `==` `affine-expr`
  // affine-constraint-conjunction ::= affine-constraint (`,`
  // affine-constraint)*

  _multi_dim_affine_expr : $ =>
      seq($._affine_expr, repeat(seq(',', $._affine_expr))),
  _affine_expr : $ => prec.right(choice(
      seq('(', $._affine_expr, ')'), seq('-', $._affine_expr),
      seq($._affine_expr, $._affine_token, $._affine_expr), $._affine_prim)),
  _affine_prim : $ =>
      choice($.integer_literal, $.value_use, $.bare_id,
             seq('symbol', '(', $.value_use, ')'),
             seq(choice('max', 'min'), '(', $._value_use_list, ')')),
  _affine_token : $ => token(
      choice('+', '-', '*', 'ceildiv', 'floordiv', 'mod', '==', '>=', '<=')),

  func_return : $ => seq(token('->'), $.type_list_attr_parens),
  func_arg_list : $ => seq(
      '(', optional(choice($.variadic, $._value_id_and_type_attr_list)), ')'),
  _value_id_and_type_attr_list : $ => seq(
      $._value_id_and_type_attr, repeat(seq(',', $._value_id_and_type_attr)),
      optional(seq(',', $.variadic))),
  _value_id_and_type_attr : $ => seq($._function_arg, optional($.attribute)),
  _function_arg : $ =>
      choice(seq($.value_use, ':', $.type), $.value_use, $.type),
  type_list_attr_parens : $ =>
      choice($.type,
             seq('(', $.type, optional($.attribute),
                 repeat(seq(',', $.type, optional($.attribute))), ')'),
             seq('(', ')')),
  variadic : $ => token('...'),

  // (func.func|llvm.func) takes arguments, an optional return type, and and
  // optional body
  _op_func : $ =>
      seq(field('visibility', optional('private')),
          field('name', $.symbol_ref_id), field('arguments', $.func_arg_list),
          field('return', optional($.func_return)),
          field('attributes', optional(seq(token('attributes'), $.attribute))),
          field('body', optional($.region))),

  // dim-use-list ::= `(` ssa-use-list? `)`
  // symbol-use-list ::= `[` ssa-use-list? `]`
  // dim-and-symbol-use-list ::= dim-use-list symbol-use-list?
  _value_use_list_parens : $ => seq('(', optional($._value_use_list), ')'),
  _dim_and_symbol_use_list : $ =>
      seq($._value_use_list_parens, optional($._dense_idx_list)),

  // assignment-list ::= assignment | assignment `,` assignment-list
  // assignment ::= ssa-value `=` ssa-value
  _value_assignment_list : $ => seq('(', optional($._value_assignment),
                                    repeat(seq(',', $._value_assignment)), ')'),
  _value_assignment : $ => seq($.value_use, '=', $.value_use),

  _dense_idx_list : $ => seq(
      '[',
      optional(seq(choice($.integer_literal, $.value_use),
                   repeat(seq(',', choice($.integer_literal, $.value_use))))),
      ']'),

  // lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list |
  // shorthand-bound
  // upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list |
  // shorthand-bound
  // shorthand-bound ::= ssa-id | `-`? integer-literal
  _bound : $ =>
      choice(seq($.attribute, $._dim_and_symbol_use_list), $._shorthand_bound),
  _shorthand_bound : $ => choice($.value_use, $.integer_literal),

  // Dialect-specific attributes
  restrict_attr : $ => token('restrict'),
  writable_attr : $ => token('writable'),
  gather_dims_attr : $ =>
      seq(token('gather_dims'), '(', $._dense_idx_list, ')'),
  scatter_dims_attr : $ =>
      seq(token('scatter_dims'), '(', $._dense_idx_list, ')'),
  unique_attr : $ => token('unique'),
  nofold_attr : $ => token('nofold'),
  outer_dims_perm_attr : $ =>
      seq(token('outer_dims_perm'), '=', $._dense_idx_list),
  inner_dims_pos_attr : $ =>
      seq(token('inner_dims_pos'), '=', $._dense_idx_list),
  inner_tiles_attr : $ => seq(token('inner_tiles'), '=', $._dense_idx_list),
  isWrite_attr : $ => token(choice('read', 'write')),
  localityHint_attr : $ => seq(token('locality'), '<', $.integer_literal, '>'),
  isDataCache_attr : $ => token(choice('data', 'instr')),
  fastmath_attr : $ =>
      seq(token('fastmath'), '<',
          seq($._fastmath_flag, repeat(seq(',', $._fastmath_flag))), '>'),
  _fastmath_flag : $ => token(choice('none', 'reassoc', 'nnan', 'ninf', 'nsz',
                                     'arcp', 'contract', 'afn', 'fast')),

  // Comment (standard BCPL)
  comment : $ => token(seq('//', /.*/)),

  // TODO: complete
  custom_operation : $ =>
      choice($.builtin_dialect, $.func_dialect, $.llvm_dialect, $.arith_dialect,
             $.math_dialect, $.cf_dialect, $.scf_dialect, $.memref_dialect,
             $.vector_dialect, $.tensor_dialect, $.bufferization_dialect,
             $.affine_dialect, $.linalg_dialect)
}

               module.exports = grammar({
  name : 'mlir',
  extras : $ => [/\s/, $.comment],
  conflicts : $ => [[ $._static_dim_list, $._static_dim_list ],
                    [ $.dictionary_attribute, $.region ]],
  rules : Object.assign(common, builtin_dialect, func_dialect, llvm_dialect,
                        arith_dialect, math_dialect, cf_dialect, scf_dialect,
                        memref_dialect, vector_dialect, tensor_dialect,
                        bufferization_dialect, affine_dialect, linalg_dialect)
});
