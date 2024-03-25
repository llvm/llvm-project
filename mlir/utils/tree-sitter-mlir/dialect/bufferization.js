'use strict';

module.exports = {
  bufferization_dialect : $ => choice(
                            seq('bufferization.alloc_tensor',
                                field('in', $._value_use_list_parens),
                                field('copy', optional(seq(token('copy'), '(',
                                                           $.value_use, ')'))),
                                field('size_hint',
                                      optional(seq(token('size_hint'), '=',
                                                   $.value_use))),
                                field('attributes', optional($.attribute)),
                                field('return', $._type_annotation)),

                            // operation ::= `bufferization.to_memref` $tensor
                            // attr-dict `:` type($memref)
                            seq('bufferization.to_memref',
                                field('tensor', $.value_use),
                                field('attributes', optional($.attribute)),
                                field('return', $._type_annotation)),

                            // operation ::= `bufferization.to_tensor` $memref
                            //               (`restrict` $restrict^)?
                            //               (`writable` $writable^)? attr-dict
                            //               `:` type($memref)
                            seq('bufferization.to_tensor',
                                field('memref', $.value_use),
                                field('restrict', optional($.restrict_attr)),
                                field('writable', optional($.writable_attr)),
                                field('attributes', optional($.attribute)),
                                field('return', $._type_annotation)))
}
