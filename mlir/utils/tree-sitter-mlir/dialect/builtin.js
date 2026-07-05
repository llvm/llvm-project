'use strict';

module.exports = {
  builtin_dialect : $ => prec.right(choice(
                      // operation ::= `builtin.module` ($sym_name^)?
                      // attr-dict-with-keyword $bodyRegion
                      seq(choice('builtin.module', 'module'),
                          field('name', optional($.bare_id)),
                          field('attributes', optional($.attribute)),
                          field('body', $.region)),

                      // operation ::= `builtin.unrealized_conversion_cast`
                      // ($inputs^ `:` type($inputs))?
                      //                `to` type($outputs) attr-dict
                      seq(choice('builtin.unrealized_conversion_cast',
                                 'unrealized_conversion_cast'),
                          field('inputs', optional($._value_use_type_list)),
                          token('to'), field('outputs', $._type_list_no_parens),
                          field('attributes', optional($.attribute)))))
}
