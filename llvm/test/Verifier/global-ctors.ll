; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, ptr, i8 } ] [
  { i32, ptr, i8 } { i32 65535, ptr null, i8 0 }
]
; CHECK: wrong type for intrinsic global variable

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr, i8 } ] [
  { i32, ptr, ptr, i8 } { i32 65535, ptr null, ptr null, i8 0}
]
; CHECK: wrong type for intrinsic global variable
