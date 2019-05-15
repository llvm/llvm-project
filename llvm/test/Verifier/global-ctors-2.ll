; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void()* } ] [
  { i32, void()* } { i32 65535, void ()* null }
]
; CHECK: the third field of the element type is mandatory, specify i8* null to migrate from the obsoleted 2-field form
