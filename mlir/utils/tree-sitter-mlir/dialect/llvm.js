'use strict';

module.exports = {
  llvm_dialect : $ => prec.right(choice(
                   seq('llvm.func', $._op_func),

                   seq('llvm.return',
                       field('attributes', optional($.attribute)),
                       field('results', optional($._value_use_type_list)))))
}
