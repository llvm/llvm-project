; RUN: llvm-as < %s | llvm-dis

define { i32 } @fooa() nounwind {
  ret i32 0
}
define { i32 } @foob() nounwind {
  ret {i32}{ i32 0 }
}
define [1 x i32] @fooc() nounwind {
  ret i32 0
}
define [1 x i32] @food() nounwind {
  ret [1 x i32][ i32 0 ]
}
