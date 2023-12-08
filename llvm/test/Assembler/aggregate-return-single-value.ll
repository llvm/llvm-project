; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s

define { i32 } @foob() nounwind {
  ret {i32}{ i32 0 }
}
define [1 x i32] @food() nounwind {
  ret [1 x i32][ i32 0 ]
}
