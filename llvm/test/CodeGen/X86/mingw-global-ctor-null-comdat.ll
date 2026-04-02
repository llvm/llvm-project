; RUN: llc -mtriple=x86_64-w64-mingw32 %s -o - | FileCheck %s

; CHECK: my_init_func

define internal void @my_init_func() {
  ret void
}

; Test that the backend gracefully handles null/invalid COMDAT keys
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @my_init_func, ptr null }
]