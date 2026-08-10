'use strict';

module.exports = {
  tensor_dialect : $ => choice(
                     // operation ::= `tensor.empty` `(`$dynamicSizes`)`
                     // attr-dict `:` type($result)
                     seq('tensor.empty',
                         field('dynamicSizes', $._value_use_list_parens),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.cast` $source attr-dict `:`
                     // type($source) `to` type($dest)
                     seq('tensor.cast', field('in', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.dim` attr-dict $source `,` $index
                     // `:` type($source)
                     seq('tensor.dim',
                         field('attributes', optional($.attribute)),
                         field('tensor', $.value_use), ',',
                         field('index', $.value_use),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.collapse_shape` $src
                     // $reassociation attr-dict `:` type($src)
                     //                `into` type($result)
                     seq(choice('tensor.collapse_shape', 'tensor.expand_shape'),
                         field('tensor', $.value_use),
                         field('reassociation', $.nested_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.extract` $tensor `[` $indices `]`
                     // attr-dict `:` type($tensor)
                     seq('tensor.extract', field('tensor', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.insert` $scalar `into` $dest `[`
                     // $indices `]` attr-dict
                     //               `:` type($dest)
                     seq('tensor.insert', field('scalar', $.value_use),
                         token('into'), field('destination', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.extract_slice` $source ``
                     //                custom<DynamicIndexList>($offsets,
                     //                $static_offsets)
                     //                custom<DynamicIndexList>($sizes,
                     //                $static_sizes)
                     //                custom<DynamicIndexList>($strides,
                     //                $static_strides) attr-dict `:`
                     //                type($source) `to` type($result)
                     seq('tensor.extract_slice', field('tensor', $.value_use),
                         field('offsets', $._dense_idx_list),
                         field('sizes', $._dense_idx_list),
                         field('strides', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.insert_slice` $source `into` $dest
                     // ``
                     //                custom<DynamicIndexList>($offsets,
                     //                $static_offsets)
                     //                custom<DynamicIndexList>($sizes,
                     //                $static_sizes)
                     //                custom<DynamicIndexList>($strides,
                     //                $static_strides) attr-dict `:`
                     //                type($source) `into` type($dest)
                     // operation ::= `tensor.parallel_insert_slice` $source
                     // `into` $dest ``
                     //                custom<DynamicIndexList>($offsets,
                     //                $static_offsets)
                     //                custom<DynamicIndexList>($sizes,
                     //                $static_sizes)
                     //                custom<DynamicIndexList>($strides,
                     //                $static_strides) attr-dict `:`
                     //                type($source) `into` type($dest)
                     seq(choice('tensor.insert_slice',
                                'tensor.parallel_insert_slice'),
                         field('source', $.value_use), token('into'),
                         field('destination', $.value_use),
                         field('offsets', $._dense_idx_list),
                         field('sizes', $._dense_idx_list),
                         field('strides', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.from_elements` $elements attr-dict
                     // `:` type($result)
                     seq('tensor.from_elements',
                         field('elements', optional($._value_use_list)),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.gather` $source `[` $indices `]`
                     //               `gather_dims` `(` $gather_dims `)`
                     //               (`unique` $unique^)?
                     //               attr-dict
                     //               `:` functional-type(operands, results)
                     seq('tensor.gather', field('source', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('gatherDims', $.gather_dims_attr),
                         field('unique', optional($.unique_attr)),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `tensor.scatter` $source `into` $dest `[`
                     // $indices `]`
                     //               `scatter_dims` `(` $scatter_dims `)`
                     //               (`unique` $unique^)?
                     //               attr-dict
                     //               `:` functional-type(operands, results)
                     seq('tensor.scatter', field('source', $.value_use),
                         token('into'), field('destination', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('scatterDims', $.scatter_dims_attr),
                         field('unique', optional($.unique_attr)),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `tensor.pad` $source
                     //               (`nofold` $nofold^)?
                     //               `low` `` custom<DynamicIndexList>($low,
                     //               $static_low) `high` ``
                     //               custom<DynamicIndexList>($high,
                     //               $static_high) $region attr-dict `:`
                     //               type($source) `to` type($result)
                     seq('tensor.pad', field('source', $.value_use),
                         field('nofold', optional($.nofold_attr)),
                         field('low', seq(token('low'), $._dense_idx_list)),
                         field('high', seq(token('high'), $._dense_idx_list)),
                         field('body', $.region),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.reshape` $source `(` $shape `)`
                     // attr-dict
                     //                `:` functional-type(operands, results)
                     seq('tensor.reshape', field('tensor', $.value_use),
                         field('shape', $._value_use_list_parens),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `tensor.splat` $input attr-dict `:`
                     // type($aggregate)
                     seq('tensor.splat', field('input', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.pack` $source
                     //               (`padding_value` `(` $padding_value^ `:`
                     //               type($padding_value) `)`)?
                     //               (`outer_dims_perm` `=` $outer_dims_perm^)?
                     //               `inner_dims_pos` `=` $inner_dims_pos
                     //               `inner_tiles` `=`
                     //               custom<DynamicIndexList>($inner_tiles,
                     //               $static_inner_tiles) `into` $dest
                     //               attr-dict `:` type($source) `->`
                     //               type($dest)
                     // operation ::= `tensor.unpack` $source
                     //               (`outer_dims_perm` `=` $outer_dims_perm^)?
                     //               `inner_dims_pos` `=` $inner_dims_pos
                     //               `inner_tiles` `=`
                     //               custom<DynamicIndexList>($inner_tiles,
                     //               $static_inner_tiles) `into` $dest
                     //               attr-dict `:` type($source) `->`
                     //               type($dest)
                     seq(choice('tensor.pack', 'tensor.unpack'),
                         field('source', $.value_use),
                         field('padding_value',
                               optional(seq(token('padding_value'), '(',
                                            $._value_use_and_type, ')'))),
                         field('outer_dims_perm',
                               optional($.outer_dims_perm_attr)),
                         field('inner_dims_pos', $.inner_dims_pos_attr),
                         field('inner_tiles', $.inner_tiles_attr),
                         token('into'), field('destination', $.value_use),
                         field('return', $._function_type_annotation)),

                     // operation ::= `tensor.generate` $dynamicExtents $body
                     // attr-dict `:` type($result)
                     seq('tensor.generate',
                         field('dynamicExtents', $._value_use_list),
                         field('body', $.region),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `tensor.rank` $tensor attr-dict `:`
                     // type($tensor) operation ::= `tensor.yield` $value
                     // attr-dict `:` type($value)
                     seq(choice('tensor.rank', 'tensor.yield'),
                         field('tensor', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)))
}
