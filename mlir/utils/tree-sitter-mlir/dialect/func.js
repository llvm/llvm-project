'use strict';

module.exports = {
  func_dialect : $ => prec.right(choice(
                   // operation ::= `func.call_indirect` $callee `(`
                   // $callee_operands `)` attr-dict
                   //               `:` type($callee)
                   // operation ::= `func.call` $callee `(` $operands `)`
                   // attr-dict
                   //               `:` functional-type($operands, results)
                   seq(choice('func.call', 'call', 'func.call_indirect',
                              'call_indirect'),
                       field('callee', $.symbol_ref_id),
                       field('operands', $._value_use_list_parens),
                       field('attributes', optional($.attribute)),
                       field('return', $._function_type_annotation)),

                   // operation ::= `func.constant` attr-dict $value `:`
                   // type(results)
                   seq(choice('func.constant', 'constant'),
                       field('attributes', optional($.attribute)),
                       field('value', $.symbol_ref_id),
                       field('return', $._function_type_annotation)),

                   seq('func.func', $._op_func),

                   seq(choice('func.return', 'return'),
                       field('attributes', optional($.attribute)),
                       field('results', optional($._value_use_type_list)))))
}
