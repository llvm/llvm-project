'use strict';

module.exports = {
  scf_dialect : $ => prec.right(choice(
                  // operation ::= `scf.condition` `(` $condition `)` attr-dict
                  // ($args^ `:` type($args))?
                  seq('scf.condition',
                      field('condition', $._value_use_list_parens),
                      field('attributes', optional($.attribute)),
                      field('arguments', optional($._value_use_type_list))),

                  seq('scf.execute_region',
                      field('return', optional($._function_return)),
                      field('body', $.region),
                      field('attributes', optional($.attribute))),

                  seq('scf.if', field('condition', $.value_use),
                      field('return', optional($._function_return)),
                      field('trueblk', $.region),
                      field('falseblk',
                            optional(seq(token('else'), $.region)))),

                  // operation ::= `scf.index_switch` $arg attr-dict (`->`
                  // type($results)^)?
                  //               custom<SwitchCases>($cases, $caseRegions)
                  //               `\n`
                  //               `` `default` $defaultRegion
                  seq('scf.index_switch', field('flag', $._value_use_and_type),
                      field('attributes', optional($.attribute)),
                      field('result', optional($._function_return)),
                      $.scf_case_label, $.region,
                      repeat(seq($.scf_case_label, $.region))),

                  // scf.for %iv = %lb to %ub step %step {
                  // ... // body
                  // }
                  seq('scf.for', field('iv', $.value_use), '=',
                      field('lb', $.value_use), token('to'),
                      field('ub', $.value_use),
                      field('step', seq(token('step'), $.value_use)),
                      field('iter_args',
                            optional(seq(token('iter_args'),
                                         $._value_assignment_list))),
                      field('return', optional($._function_return)),
                      field('body', $.region),
                      field('attributes', optional($.attribute))),

                  seq('scf.forall', field('iv', $._value_use_list_parens),
                      field('bounds',
                            seq(choice(seq('=', $._value_use_list_parens,
                                           token('to')),
                                       token('in')),
                                $._value_use_list_parens)),
                      field('step', optional(seq(token('step'),
                                                 $._value_use_list_parens))),
                      field('shared_outs',
                            optional(seq(token('shared_outs'),
                                         $._value_assignment_list))),
                      field('return', optional($._function_return)),
                      field('body', $.region),
                      field('attributes', optional($.attribute))),

                  seq('scf.forall.in_parallel', field('body', $.region),
                      field('attributes', optional($.attribute))),

                  seq('scf.parallel', field('iv', $._value_use_list_parens),
                      '=', field('lb', $._value_use_list_parens), token('to'),
                      field('ub', $._value_use_list_parens),
                      field('step',
                            seq(token('step'), $._value_use_list_parens)),
                      field('init', optional(seq(token('init'),
                                                 $._value_use_list_parens))),
                      field('return', optional($._function_return)),
                      field('body', $.region),
                      field('attributes', optional($.attribute))),

                  seq('scf.reduce', field('operand', $._value_use_list_parens),
                      field('return', $._type_annotation),
                      field('body', $.region)),

                  // operation ::= `scf.reduce.return` $result attr-dict `:`
                  // type($result)
                  seq('scf.reduce.return', field('result', $.value_use),
                      field('attributes', optional($.attribute)),
                      field('return', $._type_annotation)),

                  // op ::= `scf.while` assignments `:` function-type region
                  // `do` region
                  //        `attributes` attribute-dict
                  // initializer ::= /* empty */ | `(` assignment-list `)`
                  seq('scf.while',
                      field('assignments', optional($._value_assignment_list)),
                      field('return', $._function_type_annotation),
                      field('condblk', $.region), 'do',
                      field('doblk', $.region),
                      field('attributes',
                            optional(seq('attributes', $.attribute)))),

                  // operation ::= `scf.yield` attr-dict ($results^ `:`
                  // type($results))?
                  seq('scf.yield', field('attributes', optional($.attribute)),
                      field('results', optional($._value_use_type_list))),
                  )),

  scf_case_label : $ => choice(seq(token('case'), $.integer_literal),
                               token('default'))
}
