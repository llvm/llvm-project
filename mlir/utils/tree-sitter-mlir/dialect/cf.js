'use strict';

module.exports = {
  cf_dialect : $ => prec.right(choice(
                 // operation ::= `cf.assert` $arg `,` $msg attr-dict
                 seq('cf.assert', field('argument', $.value_use), ',',
                     field('message', $.string_literal),
                     field('attributes', optional($.attribute))),

                 // operation ::= `cf.br` $dest (`(` $destOperands^ `:`
                 // type($destOperands) `)`)? attr-dict
                 seq('cf.br', field('successor', $.successor),
                     field('attributes', optional($.attribute))),

                 // operation ::= `cf.cond_br` $condition `,`
                 //               $trueDest(`(` $trueDestOperands ^ `:`
                 //               type($trueDestOperands)`)`)? `,`
                 //               $falseDest(`(` $falseDestOperands ^ `:`
                 //               type($falseDestOperands)`)`)? attr-dict
                 seq('cf.cond_br', field('condition', $.value_use), ',',
                     field('trueblk', $.successor), ',',
                     field('falseblk', $.successor),
                     field('attributes', optional($.attribute))),

                 // operation ::= `cf.switch` $flag `:` type($flag) `,` `[` `\n`
                 //               custom<SwitchOpCases>(ref(type($flag)),$defaultDestination,
                 //               $defaultOperands,
                 //               type($defaultOperands),
                 //               $case_values,
                 //               $caseDestinations,
                 //               $caseOperands,
                 //               type($caseOperands))
                 //               `]`
                 //               attr-dict
                 seq('cf.switch', field('flag', $._value_use_and_type), ',',
                     '[', $.cf_case_label, $.successor,
                     repeat(seq(',', $.cf_case_label, $.successor)), ']',
                     field('attributes', optional($.attribute))),
                 )),

  cf_case_label : $ => seq(choice($.integer_literal, token('default')), ':')
}
