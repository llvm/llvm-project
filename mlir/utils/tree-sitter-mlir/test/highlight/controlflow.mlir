func.func @simple(i64, i1) -> i64 {
// <- function.builtin
//        ^ function
//               ^ punctuation.bracket
//                ^ type.builtin
//                   ^ punctuation.delimeter
//                     ^ type.builtin
//                       ^ punctuation.bracket
//                         ^ operator
//                            ^ type.builtin
//                                ^ punctuation.bracket
^bb0(%a: i64, %cond: i1):
// <- tag
//   ^ variable.parameter
//       ^ type.builtin
//            ^ variable.parameter
//                   ^ type.builtin
  cf.cond_br %cond, ^bb1, ^bb2
// ^ function.builtin
//           ^ variable.parameter
//                  ^ tag
//                        ^ tag

^bb1:
// <- tag
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument
// ^ function.builtin
//      ^ tag
//           ^ variable.parameter
//               ^ type.builtin
//                       ^ comment

^bb2:
// <- tag
  %b = arith.addi %a, %a : i64
// ^ variable
//   ^ operator
//     ^ function.builtin
//                ^ variable.parameter
//                    ^ variable.parameter
//                         ^ type.builtin
  cf.br ^bb3(%b: i64)    // Branch passes %b as the argument
// ^ function.builtin
//      ^ tag
//           ^ variable
//               ^ type.builtin
//                       ^ comment
^bb3(%c: i64):
// <- tag
//   ^ variable.parameter
//        ^ type.builtin
  cf.br ^bb4(%c, %a : i64, i64)
// ^ function.builtin
//      ^ tag
//           ^ variable.parameter
//               ^ variable.parameter
//                    ^ type.builtin
//                         ^ type.builtin
^bb4(%d : i64, %e : i64):
// <- tag
//   ^ variable.parameter
//        ^ type.builtin
//             ^ variable.parameter
//                  ^ type.builtin
  %0 = arith.addi %d, %e : i64
// ^ variable
//   ^ operator
//     ^ function.builtin
//                ^ variable.parameter
//                    ^ variable.parameter
//                          ^ type.builtin
  return %0 : i64   // Return is also a terminator.
// ^ function.builtin
//       ^ variable
//            ^ type.builtin
//                  ^ comment
}
// <- punctuation.bracket
