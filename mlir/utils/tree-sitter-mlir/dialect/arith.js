'use strict';

module.exports = {
  arith_dialect : $ => choice(
                    // operation ::= `arith.constant` attr-dict $value
                    seq('arith.constant',
                        field('attributes', optional($.attribute)),
                        field('value', $._literal_and_type)),

                    // operation ::= `arith.addi` $lhs `,` $rhs attr-dict `:`
                    // type($result) operation ::= `arith.subi` $lhs `,` $rhs
                    // attr-dict `:` type($result) operation ::= `arith.divsi`
                    // $lhs `,` $rhs attr-dict `:` type($result) operation ::=
                    // `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
                    // operation ::= `arith.ceildivsi` $lhs `,` $rhs attr-dict
                    // `:` type($result) operation ::= `arith.ceildivui` $lhs
                    // `,` $rhs attr-dict `:` type($result) operation ::=
                    // `arith.floordivsi` $lhs `,` $rhs attr-dict `:`
                    // type($result) operation ::= `arith.remsi` $lhs `,` $rhs
                    // attr-dict `:` type($result) operation ::= `arith.remui`
                    // $lhs `,` $rhs attr-dict `:` type($result) operation ::=
                    // `arith.muli` $lhs `,` $rhs attr-dict `:` type($result)
                    // operation ::= `arith.mulsi_extended` $lhs `,` $rhs
                    // attr-dict `:` type($lhs) operation ::=
                    // `arith.mului_extended` $lhs `,` $rhs attr-dict `:`
                    // type($lhs) operation ::= `arith.andi` $lhs `,` $rhs
                    // attr-dict `:` type($result) operation ::= `arith.ori`
                    // $lhs `,` $rhs attr-dict `:` type($result) operation ::=
                    // `arith.xori` $lhs `,` $rhs attr-dict `:` type($result)
                    // operation ::= `arith.maxsi` $lhs `,` $rhs attr-dict `:`
                    // type($result) operation ::= `arith.maxui` $lhs `,` $rhs
                    // attr-dict `:` type($result) operation ::= `arith.minsi`
                    // $lhs `,` $rhs attr-dict `:` type($result) operation ::=
                    // `arith.minui` $lhs `,` $rhs attr-dict `:` type($result)
                    // operation ::= `arith.shli` $lhs `,` $rhs attr-dict `:`
                    // type($result) operation ::= `arith.shrsi` $lhs `,` $rhs
                    // attr-dict `:` type($result) operation ::= `arith.shrui`
                    // $lhs `,` $rhs attr-dict `:` type($result)
                    seq(choice('arith.addi', 'arith.subi', 'arith.divsi',
                               'arith.divui', 'arith.ceildivsi',
                               'arith.ceildivui', 'arith.floordivsi',
                               'arith.remsi', 'arith.remui', 'arith.muli',
                               'arith.mulsi_extended', 'arith.mului_extended',
                               'arith.andi', 'arith.ori', 'arith.xori',
                               'arith.maxsi', 'arith.maxui', 'arith.minsi',
                               'arith.minui', 'arith.shli', 'arith.shrsi',
                               'arith.shrui'),
                        field('lhs', $.value_use), ',',
                        field('rhs', $.value_use),
                        field('attributes', optional($.attribute)),
                        field('return', $._type_annotation)),

                    // operation ::= `arith.addui_extended` $lhs `,` $rhs
                    // attr-dict `:` type($sum)
                    //                `,` type($overflow)
                    seq('arith.addui_extended', field('lhs', $.value_use), ',',
                        field('rhs', $.value_use),
                        field('attributes', optional($.attribute)),
                        field('return', seq(':', $.type, ',', $.type))),

                    // operation ::= `arith.addf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.divf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.maxf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.minf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.mulf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.remf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    // operation ::= `arith.subf` $lhs `,` $rhs (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    seq(choice('arith.addf', 'arith.divf', 'arith.maxf',
                               'arith.minf', 'arith.mulf', 'arith.remf',
                               'arith.subf'),
                        field('lhs', $.value_use), ',',
                        field('rhs', $.value_use),
                        field('fastmath', optional($.fastmath_attr)),
                        field('attributes', optional($.attribute)),
                        field('return', $._type_annotation)),

                    // operation ::= `arith.negf` $operand (`fastmath` ``
                    // $fastmath^)?
                    //                attr-dict `:` type($result)
                    seq(choice('arith.negf'), field('operand', $.value_use),
                        field('fastmath', optional($.fastmath_attr)),
                        field('attributes', optional($.attribute)),
                        field('return', $._type_annotation)),

                    // operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs
                    // attr-dict `:` type($lhs) operation ::= `arith.cmpf`
                    // $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
                    seq(choice('arith.cmpi', 'arith.cmpf'),
                        field('predicate',
                              choice('eq', 'ne', 'oeq', 'olt', 'ole', 'ogt',
                                     'oge', 'slt', 'sle', 'sgt', 'sge', 'ult',
                                     'ule', 'ugt', 'uge', $.string_literal)),
                        ',', field('lhs', $.value_use), ',',
                        field('rhs', $.value_use),
                        field('attributes', optional($.attribute)),
                        field('return', $._type_annotation)),

                    // operation ::= `arith.extf` $in attr-dict `:` type($in)
                    // `to` type($out) operation ::= `arith.extsi` $in attr-dict
                    // `:` type($in) `to` type($out) operation ::= `arith.extui`
                    // $in attr-dict `:` type($in) `to` type($out) operation ::=
                    // `arith.fptosi` $in attr-dict `:` type($in) `to`
                    // type($out) operation ::= `arith.fptoui` $in attr-dict `:`
                    // type($in) `to` type($out) operation ::=
                    // `arith.index_cast` $in attr-dict `:` type($in) `to`
                    // type($out) operation ::= `arith.index_castui` $in
                    // attr-dict `:` type($in) `to` type($out) operation ::=
                    // `arith.sitofp` $in attr-dict `:` type($in) `to`
                    // type($out) operation ::= `arith.uitofp` $in attr-dict `:`
                    // type($in) `to` type($out) operation ::= `arith.bitcast`
                    // $in attr-dict `:` type($in) `to` type($out) operation ::=
                    // `arith.truncf` $in attr-dict `:` type($in) `to`
                    // type($out) operation ::= `arith.trunci` $in attr-dict `:`
                    // type($in) `to` type($out)
                    seq(choice('arith.extf', 'arith.extsi', 'arith.extui',
                               'arith.fptosi', 'arith.fptoui',
                               'arith.index_cast', 'arith.index_castui',
                               'arith.sitofp', 'arith.uitofp', 'arith.bitcast',
                               'arith.truncf', 'arith.trunci'),
                        field('in', $.value_use),
                        field('attributes', optional($.attribute)),
                        field('return', $._type_annotation)),

                    seq('arith.select', field('cond', $.value_use), ',',
                        field('trueblk', $.value_use), ',',
                        field('falseblk', $.value_use),
                        field('return', $._type_annotation)))
}
