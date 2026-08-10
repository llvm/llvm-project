'use strict';

module.exports = {
  affine_dialect : $ => prec.right(choice(
                     seq('affine.apply',
                         field('operand',
                               seq($.attribute, $._dim_and_symbol_use_list)),
                         field('attributes', optional($.attribute))),

                     // operation ::= `affine.delinearize_index` $linear_index
                     // `into` ` `
                     //               `(` $basis `)` attr-dict `:`
                     //               type($multi_index)
                     seq('affine.delinearlize_index',
                         field('operand', $.value_use), 'into',
                         field('basis', $._value_use_list_parens),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `affine.dma_start` ssa-use `[`
                     // multi-dim-affine-map-of-ssa-ids `]`,
                     //               `[` multi-dim-affine-map-of-ssa-ids `]`,
                     //               `[` multi-dim-affine-map-of-ssa-ids `]`,
                     //               ssa-use `:` memref-type
                     seq(choice('affine.dma_start', 'affine.dma_wait'),
                         field('operands',
                               seq($.value_use, $._multi_dim_affine_expr_sq,
                                   repeat(seq(',', $.value_use,
                                              $._multi_dim_affine_expr_sq)))),
                         ',', field('numElements', $._value_use_list),
                         field('return', $._type_annotation)),

                     // operation   ::= `affine.for` ssa-id `=` lower-bound `to`
                     // upper-bound
                     //                 (`step` integer-literal)? `{` op* `}`
                     seq('affine.for', field('iv', $.value_use), '=',
                         field('lowerBound',
                               seq(optional(token('max')), $._bound)),
                         token('to'),
                         field('upperBound',
                               seq(optional(token('min')), $._bound)),
                         field('step',
                               optional(seq(token('step'), $.integer_literal))),
                         field('iter_args',
                               optional(seq(token('iter_args'),
                                            $._value_assignment_list))),
                         field('return', optional($._function_return)),
                         field('body', $.region),
                         field('attributes', optional($.attribute))),

                     // operation  ::= `affine.if` if-op-cond `{` op* `}`
                     // (`else` `{` op* `}`)? if-op-cond ::= integer-set-attr
                     // dim-and-symbol-use-list
                     seq('affine.if',
                         field('condition',
                               seq($.attribute, $._dim_and_symbol_use_list)),
                         field('return', optional($._function_return)),
                         field('trueblk', $.region),
                         field('falseblk',
                               optional(seq(token('else'), $.region))),
                         field('attributes', optional($.attribute))),

                     // operation ::= `affine.load` ssa-use `[`
                     // multi-dim-affine-map-of-ssa-ids `]`
                     //               `:` memref-type
                     seq(choice('affine.load', 'affine.vector_load'),
                         field('operand', $.value_use),
                         field('multiDimAffineMap',
                               $._multi_dim_affine_expr_sq),
                         field('return', $._type_annotation)),

                     // operation ::= `affine.min` affine-map-attribute
                     // dim-and-symbol-use-list
                     seq(choice('affine.min', 'affine.max'),
                         field('operand',
                               seq($.attribute, $._dim_and_symbol_use_list))),

                     seq('affine.parallel',
                         field('iv', $._value_use_list_parens), '=',
                         field('lowerBound', $._multi_dim_affine_expr_parens),
                         token('to'),
                         field('upperBound', $._multi_dim_affine_expr_parens),
                         field('step',
                               optional(seq(token('step'),
                                            $._multi_dim_affine_expr_parens))),
                         field('reduce',
                               optional(seq(token('reduce'), '(',
                                            $.string_literal,
                                            repeat(seq(',', $.string_literal)),
                                            ')'))),
                         field('return', optional($._function_return)),
                         field('body', $.region)),

                     seq('affine.prefetch', field('source', $.value_use),
                         field('indices',
                               optional($._multi_dim_affine_expr_sq)),
                         ',', field('isWrite', $.isWrite_attr), ',',
                         field('localityHint', $.localityHint_attr), ',',
                         field('isDataCache', $.isDataCache_attr),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `affine.store` ssa-use, ssa-use `[`
                     // multi-dim-affine-map-of-ssa-ids `]`
                     //               `:` memref-type
                     seq(choice('affine.store', 'affine.vector_store'),
                         field('source', $.value_use), ',',
                         field('destination', $.value_use),
                         field('multiDimAffineMap',
                               $._multi_dim_affine_expr_sq),
                         field('return', $._type_annotation)),

                     // operation ::= `affine.yield` attr-dict ($operands^ `:`
                     // type($operands))?
                     seq('affine.yield',
                         field('attributes', optional($.attribute)),
                         field('results', optional($._value_use_type_list)))))
}
