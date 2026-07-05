'use strict';

module.exports = {
  memref_dialect : $ => choice(
                     // operation ::= `memref.assume_alignment` $memref `,`
                     // $alignment attr-dict `:` type($memref)
                     seq('memref.assume_alignment',
                         field('memref', $.value_use), ',',
                         field('alignment', $.integer_literal),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.alloc` `(`$dynamicSizes`)` (`` `[`
                     // $symbolOperands^ `]`)? attr-dict
                     //               `:` type($memref)
                     // operation ::= `memref.alloca` `(`$dynamicSizes`)` (``
                     // `[` $symbolOperands^ `]`)? attr-dict
                     //               `:` type($memref)
                     seq(choice('memref.alloc', 'memref.alloca'),
                         field('dynamicSizes', $._value_use_list_parens),
                         field('symbolOperands', optional($._dense_idx_list)),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.cast` $source attr-dict `:`
                     // type($source) `to` type($dest)
                     seq('memref.cast', field('in', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.copy` $source `,` $target
                     // attr-dict
                     //               `:` type($source) `to` type($target)
                     seq('memref.copy', field('source', $.value_use), ',',
                         field('target', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.collapse_shape` $src
                     // $reassociation attr-dict
                     //               `:` type($src) `into` type($result)
                     // operation ::= `memref.expand_shape` $src $reassociation
                     // attr-dict
                     //               `:` type($src) `into` type($result)
                     seq(choice('memref.collapse_shape', 'memref.expand_shape'),
                         field('source', $.value_use),
                         field('reassociation', $.nested_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.dealloc` $memref attr-dict `:`
                     // type($memref)
                     seq('memref.dealloc', field('memref', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.dim` attr-dict $source `,` $index
                     // `:` type($source)
                     seq('memref.dim',
                         field('attributes', optional($.attribute)),
                         field('source', $.value_use), ',',
                         field('index', $.value_use),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.load` $memref `[` $indices `]`
                     // attr-dict `:` type($memref)
                     seq('memref.load',
                         field('memref', seq($.value_use, $._dense_idx_list)),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     seq('memref.prefetch', field('source', $.value_use),
                         field('indices', optional($._dense_idx_list)), ',',
                         field('isWrite', $.isWrite_attr), ',',
                         field('localityHint', $.localityHint_attr), ',',
                         field('isDataCache', $.isDataCache_attr),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.rank` $memref attr-dict `:`
                     // type($memref)
                     seq('memref.rank', field('memref', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.realloc` $source (`(`
                     // $dynamicResultSize^ `)`)? attr-dict
                     //               `:` type($source) `to` type(results)
                     seq('memref.realloc', field('source', $.value_use),
                         field('dynamicResultSize',
                               optional($._value_use_list_parens)),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.reinterpret_cast` $source `to`
                     // `offset` `` `:`
                     //               custom<DynamicIndexList>($offsets,
                     //               $static_offsets)
                     //               `` `,` `sizes` `` `:`
                     //               custom<DynamicIndexList>($sizes,
                     //               $static_sizes)
                     //               `` `,` `strides` `` `:`
                     //               custom<DynamicIndexList>($strides,
                     //               $static_strides) attr-dict `:`
                     //               type($source) `to` type($result)
                     seq('memref.reinterpret_cast',
                         field('source', $.value_use), token('to'),
                         field('offset', seq(token('offset'), ':',
                                             $._dense_idx_list, ',')),
                         field('sizes', seq(token('sizes'), ':',
                                            $._dense_idx_list, ',')),
                         field('strides',
                               seq(token('strides'), ':', $._dense_idx_list)),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.reshape` $source `(` $shape `)`
                     // attr-dict
                     //               `:` functional-type(operands, results)
                     seq('memref.reshape', field('source', $.value_use),
                         field('shape', seq('(', $.value_use, ')')),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `memref.store` $value `,` $memref `[`
                     // $indices `]` attr-dict
                     //                `:` type($memref)
                     seq('memref.store', field('source', $.value_use), ',',
                         field('destination', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.subview` $source ``
                     //               custom<DynamicIndexList>($offsets,
                     //               $static_offsets)
                     //               custom<DynamicIndexList>($sizes,
                     //               $static_sizes)
                     //               custom<DynamicIndexList>($strides,
                     //               $static_strides) attr-dict `:`
                     //               type($source) `to` type($result)
                     seq('memref.subview', field('source', $.value_use),
                         field('offsets', $._dense_idx_list),
                         field('sizes', $._dense_idx_list),
                         field('strides', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `memref.view` $source `[` $byte_shift `]`
                     // `` `[` $sizes `]` attr-dict
                     //         `:` type($source) `to` type(results)
                     seq('memref.view', field('source', $.value_use),
                         field('byte_shift', $._dense_idx_list),
                         field('sizes', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)))
}
