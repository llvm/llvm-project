module.exports = grammar({
  name : 'mlir',
  extras : $ => [/\s/,
                 $.comment,
],
  conflicts : $ => [],
  rules : {
    // Top level production:
    //   (operation | attribute-alias-def | type-alias-def)
    toplevel : $ => seq(choice(
                 $.operation,
                 $.attribute_alias_def,
                 $.type_alias_def,
                 )),

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
    _hex_digit : $ => /[0-9a-fA-F]/,
    integer_literal : $ => choice($._decimal_literal, $._hexadecimal_literal),
    _decimal_literal : $ => repeat1($._digit),
    _hexadecimal_literal : $ => seq('0x', repeat1($._hex_digit)),
    float_literal : $ => token(
                      seq(optional(/[-+]/), repeat1(/[0_9]/),
                          optional(seq('.', repeat(/[0-9]/),
                                       optional(seq(/[eE]/, optional(/[-+]/),
                                                    repeat1(/[0-9]/))))))),
    string_literal : $ => seq(
                       '"',
                       repeat(token.immediate(prec(1, /[^\\"\n\f\v\r]+/))),
                       '"',
                       ),

    // Identifiers
    //   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    //   bare-id-list ::= bare-id (`,` bare-id)*
    //   value-id ::= `%` suffix-id
    //   suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
    //   alias-name :: = bare-id
    //
    //   symbol-ref-id ::= `@` (suffix-id | string-literal) (`::`
    //   symbol-ref-id)?
    //   value-id-list ::= value-id (`,` value-id)*
    //
    //   // Uses of value, e.g. in an operand list to an operation.
    //   value-use ::= value-id
    //   value-use-list ::= value-use (`,` value-use)*
    bare_id : $ => seq(token(/[a-zA-Z_]/),
                       token.immediate(repeat(/[a-zA-Z0-9_$]/))),
    bare_id_list : $ => seq($.bare_id, repeat(seq(',', $.bare_id))),
    value_id : $ => seq('%', $._suffix_id),
    alias_name : $ => $.bare_id,
    _suffix_id : $ => choice(repeat1(/[0-9]/),
                             seq(/[a-zA-Z_$.]/, repeat(/[a-zA-Z0-9_$.]/))),
    symbol_ref_id : $ => seq('@', choice($._suffix_id, $.string_literal),
                             optional(seq('::', $.symbol_ref_id))),
    value_use : $ => $.value_id,
    value_use_list : $ => seq($.value_use, repeat(seq(',', $.value_use))),

    // Operations
    //   operation            ::= op-result-list? (generic-operation |
    //                            custom-operation)
    //                            trailing-location?
    //   generic-operation    ::= string-literal `(` value-use-list? `)`
    //   successor-list?
    //                            region-list? dictionary-attribute? `:`
    //                            function-type
    //   custom-operation     ::= bare-id custom-operation-format
    //   op-result-list       ::= op-result (`,` op-result)* `=`
    //   op-result            ::= value-id (`:` integer-literal)
    //   successor-list       ::= `[` successor (`,` successor)* `]`
    //   successor            ::= caret-id (`:` bb-arg-list)?
    //   region-list          ::= `(` region (`,` region)* `)`
    //   dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)?
    //                            `}`
    //   trailing-location    ::= (`loc` `(` location `)`)?
    operation : $ => seq(optional($.op_result_list),
                         choice($.generic_operation, $.custom_operation),
                         optional($.trailing_location)),
    generic_operation : $ =>
                          seq($.string_literal, '(', optional($.value_use_list),
                              ')', optional($.successor_list),
                              optional($.region_list),
                              optional($.dictionary_attribute), ':',
                              $.function_type),
    // custom-operation rule is defined later in the grammar, post the generic.
    op_result_list : $ => seq($.op_result, repeat(seq(',', $.op_result)), '='),
    op_result : $ => seq($.value_id, optional(seq(':', $.integer_literal))),
    successor_list : $ => seq('[', $.successor, repeat(seq(',', $.successor)),
                              ']'),
    successor : $ => seq($.caret_id, optional(seq(':', $.block_arg_list))),
    region_list : $ => seq('(', $.region, repeat(seq(',', $.region)), ')'),
    dictionary_attribute : $ => seq(
                             '{',
                             optional(seq($.attribute_entry,
                                          repeat(seq(',', $.attribute_entry)))),
                             '}'),
    trailing_location : $ => seq('loc(', $.location, ')'),
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
    value_id_and_type : $ => seq($.value_id, ':', $.type),
    value_id_and_type_list : $ => seq($.value_id_and_type,
                                      repeat(seq(',', $.value_id_and_type))),
    block_arg_list : $ => seq('(', optional($.value_id_and_type_list), ')'),

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
    //   type-list-parens)
    type : $ => choice($.type_alias, $.dialect_type, $.builtin_type),
    type_list_no_parens : $ => seq($.type, repeat(seq(',', $.type))),
    type_list_parens : $ => seq('(', optional($.type_list_no_parens), ')'),
    ssa_use_and_type : $ => seq($.ssa_use, ':', $.type),
    ssa_use : $ => $.value_use,
    ssa_use_and_type_list : $ => seq($.ssa_use_and_type,
                                     repeat(seq(',', $.ssa_use_and_type))),
    function_type : $ => seq(choice($.type, $.type_list_parens), '->',
                             choice($.type, $.type_list_parens)),

    // Type aliases
    //   type-alias-def ::= '!' alias-name '=' 'type' type
    //   type-alias ::= '!' alias-name
    type_alias_def : $ => seq('!', $.alias_name, '=', 'type', $.type),
    type_alias : $ => seq('!', $.alias_name),

    // Dialect Types
    //   dialect-namespace ::= bare-id
    //
    //   opaque-dialect-item ::= dialect-namespace '<' string-literal '>'
    //
    //   pretty-dialect-item ::= dialect-namespace '.'
    //   pretty-dialect-item-lead-ident
    //                                                 pretty-dialect-item-body?
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
    dialect_type : $ => seq(
                     '!', choice($.opaque_dialect_item, $.pretty_dialect_item)),
    dialect_namespace : $ => $.bare_id,
    opaque_dialect_item : $ => seq($.dialect_namespace, '<', $.string_literal,
                                   '>'),
    pretty_dialect_item : $ => seq($.dialect_namespace, '.',
                                   $.pretty_dialect_item_lead_ident,
                                   optional($.pretty_dialect_item_body)),
    pretty_dialect_item_lead_ident : $ => $.bare_id,
    pretty_dialect_item_body : $ => seq('<',
                                        repeat1($.pretty_dialect_item_contents),
                                        '>'),
    // TODO: not sure why prec.left (setting left-associated parsing) needed
    // here,
    // left-associated way avoids an ambiguity flagged by generator. It may not
    // be needed and be only papering over an issue.
    pretty_dialect_item_contents : $ => prec.left(choice(
                                     $.pretty_dialect_item_body,
                                     seq('(',
                                         repeat1(
                                             $.pretty_dialect_item_contents),
                                         ')'),
                                     seq('[',
                                         repeat1(
                                             $.pretty_dialect_item_contents),
                                         ']'),
                                     seq('{',
                                         repeat1(
                                             $.pretty_dialect_item_contents),
                                         '}'),
                                     repeat1(/[^\[<({>\])}\\0]/))),
    dialect_type : $ => seq(
                     '!', choice($.opaque_dialect_item, $.pretty_dialect_item)),

    // Builtin types
    builtin_type : $ => choice(
                     // TODO: Add builtin types
                     seq('i', repeat1(/[0-9]/))),

    // Attributes
    //   attribute-entry ::= (bare-id | string-literal) `=` attribute-value
    //   attribute-value ::= attribute-alias | dialect-attribute |
    //   builtin-attribute
    attribute_entry : $ => seq(choice($.bare_id, $.string_literal), '=',
                               $.attribute_value),
    attribute_value : $ => choice($.attribute_alias, $.dialect_attribute,
                                  $.builtin_attribute),

    // Attribute Value Aliases
    //   attribute-alias-def ::= '#' alias-name '=' attribute-value
    //   attribute-alias ::= '#' alias-name
    attribute_alias_def : $ => seq('#', $.alias_name, '=', $.attribute_value),
    attribute_alias : $ => seq('#', $.alias_name),

    // Dialect Attribute Values
    dialect_attribute : $ => seq('#', choice($.opaque_dialect_item,
                                             $.pretty_dialect_item)),

    // Builtin Attribute Values
    builtin_attribute : $ => choice(
                          // TODO
                          $.function_type,
                          $.string_literal,
                          ),

    // Comment (standard BCPL)
    comment : $ => token(seq('//', /.*/)),

    custom_operation : $ => choice(
                         // TODO: Just basic/incomplete instance.
                         seq('func', field('name', $.symbol_ref_id),
                             $.block_arg_list, '->', $.type, $.region),
                         ),
  }
});
