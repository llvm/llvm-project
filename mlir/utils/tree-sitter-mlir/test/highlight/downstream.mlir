func.func @sort_memref(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>,
// <- function.builtin
//        ^ function
//                     ^ variable.parameter
//                              ^ type.builtin
                       %init1: memref<?x?xf32>, %init2: memref<?x?xi32>) {
  thlo.sort
      ins(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>)
//    ^ keyword
//                                  ^ variable.parameter
      outs(%init1: memref<?x?xf32>, %init2: memref<?x?xi32>)
//    ^ keyword
//                                  ^ variable.parameter
      { dimension = 0 : i64, is_stable = true }
//                                       ^ constant.builtin
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
//            ^ function.builtin
//                       ^ keyword
//                            ^ variable
//                                  ^ variable
//                                        ^ type.builtin
        thlo.yield %gt : i1
      }
  func.return
// ^ function.builtin
}
