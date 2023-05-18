'use strict';

module.exports = {
  vector_dialect : $ => prec.right(choice(
                     // operation ::= `vector.bitcast` $source attr-dict `:`
                     // type($source) `to` type($result) operation ::=
                     // `vector.broadcast` $source attr-dict `:` type($source)
                     // `to` type($vector) operation ::= `vector.shape_cast`
                     // $source attr-dict `:` type($source) `to` type($result)
                     // operation ::= `vector.type_cast` $memref attr-dict `:`
                     // type($memref) `to` type($result)
                     seq(choice('vector.bitcast', 'vector.broadcast',
                                'vector.shape_cast', 'vector.type_cast'),
                         field('in', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.constant_mask` $mask_dim_sizes
                     // attr-dict `:` type(results)
                     seq('vector.constant_mask',
                         field('mask', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.create_mask` $operands attr-dict
                     // `:` type(results)
                     seq('vector.create_mask',
                         field('operands', $._value_use_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.extract` $vector `` $position
                     // attr-dict `:` type($vector) operation ::= `vector.load`
                     // $base `[` $indices `]` attr-dict
                     //               `:` type($base) `,` type($nresult)
                     // operation ::= `vector.scalable.extract` $source `[` $pos
                     // `]` attr-dict
                     //               `:` type($res) `from` type($source)
                     seq(choice('vector.extract', 'vector.load',
                                'vector.scalable.extract'),
                         field('operand', $.value_use),
                         field('indices', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.fma` $lhs `,` $rhs `,` $acc
                     // attr-dict `:` type($lhs)
                     seq('vector.fma', field('lhs', $.value_use), ',',
                         field('rhs', $.value_use), ',',
                         field('acc', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.flat_transpose` $matrix attr-dict
                     // `:` type($matrix) `->` type($res)
                     seq('vector.flat_transpose', field('matrix', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `vector.insert` $source `,` $dest
                     // $position attr-dict
                     //               `:` type($source) `into` type($dest)
                     // operation ::= `vector.scalable.insert` $source `,` $dest
                     // `[` $pos `]` attr-dict
                     //               `:` type($source) `into` type($dest)
                     // operation ::= `vector.shuffle` operands $mask attr-dict
                     // `:` type(operands) operation ::= `vector.store`
                     // $valueToStore `,` $base `[` $indices `]` attr-dict
                     //               `:` type($base) `,` type($valueToStore)
                     seq(choice('vector.insert', 'vector.scalable.insert',
                                'vector.shuffle', 'vector.store'),
                         field('source', $.value_use), ',',
                         field('destination', $.value_use),
                         field('position', $._dense_idx_list),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.insert_strided_slice` $source `,`
                     // $dest attr-dict
                     //               `:` type($source) `into` type($dest)
                     seq('vector.insert_strided_slice',
                         field('source', $.value_use), ',',
                         field('destination', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.matrix_multiply` $lhs `,` $rhs
                     // attr-dict
                     //                `:` `(` type($lhs) `,` type($rhs) `)`
                     //                `->` type($res)
                     seq('vector.matrix_multiply', field('lhs', $.value_use),
                         ',', field('rhs', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._function_type_annotation)),

                     // operation ::= `vector.print` $source attr-dict `:`
                     // type($source)
                     seq(choice('vector.print', 'vector.splat'),
                         field('operand', $.value_use),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     seq('vector.transfer_read',
                         field('source', seq($.value_use, $._dense_idx_list)),
                         field('paddingMask', repeat(seq(',', $.value_use))),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     seq('vector.transfer_write', field('vector', $.value_use),
                         ',',
                         field('source', seq($.value_use, $._dense_idx_list)),
                         field('mask', optional(seq(',', $.value_use))),
                         field('attributes', optional($.attribute)),
                         field('return', $._type_annotation)),

                     // operation ::= `vector.yield` attr-dict ($operands^ `:`
                     // type($operands))?
                     seq('vector.yield',
                         field('attributes', optional($.attribute)),
                         field('results', optional($._value_use_type_list)))))
}
