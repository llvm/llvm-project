; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @test_too_few_arguments(i32 %arg, ...) "modular-format"="printf,1,2,basic_mod" {
  ret void
}
; CHECK: modular-format attribute requires at least 5 arguments
; CHECK-NEXT: ptr @test_too_few_arguments

define void @test_first_arg_index_not_integer(i32 %arg, ...) "modular-format"="printf,1,foo,basic_mod,basic_impl" {
  ret void
}
; CHECK: modular-format attribute first arg index is not an integer
; CHECK-NEXT: ptr @test_first_arg_index_not_integer

define void @test_first_arg_index_zero(i32 %arg) "modular-format"="printf,1,0,basic_mod,basic_impl" {
  ret void
}
; CHECK: modular-format attribute first arg index is out of bounds
; CHECK-NEXT: ptr @test_first_arg_index_zero

define void @test_first_arg_index_out_of_bounds(i32 %arg) "modular-format"="printf,1,2,basic_mod,basic_impl" {
  ret void
}
; CHECK: modular-format attribute first arg index is out of bounds
; CHECK-NEXT: ptr @test_first_arg_index_out_of_bounds

define void @test_first_arg_index_out_of_bounds_varargs(i32 %arg, ...) "modular-format"="printf,1,3,basic_mod,basic_impl" {
  ret void
}
; CHECK: modular-format attribute first arg index is out of bounds
; CHECK-NEXT: ptr @test_first_arg_index_out_of_bounds_varargs

; CHECK-NOT: ptr @test_first_arg_index_in_bounds
define void @test_first_arg_index_in_bounds(i32 %arg) "modular-format"="printf,1,1,basic_mod,basic_impl" {
  ret void
}

; CHECK-NOT: ptr @test_first_arg_index_in_bounds_varargs
define void @test_first_arg_index_in_bounds_varargs(i32 %arg, ...) "modular-format"="printf,1,2,basic_mod,basic_impl" {
  ret void
}
