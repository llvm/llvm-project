; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that the "returned" attribute is optimized effectively.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: _Z3foov:
; CHECK-NEXT: .functype _Z3foov () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 1{{$}}
; CHECK-NEXT: {{^}} call      $push1=, _Znwm, $pop0{{$}}
; CHECK-NEXT: {{^}} call      $push2=, _ZN5AppleC1Ev, $pop1{{$}}
; CHECK-NEXT: return    $pop2{{$}}
%class.Apple = type { i8 }
declare noalias ptr @_Znwm(i32)
declare ptr @_ZN5AppleC1Ev(ptr returned)
define ptr @_Z3foov() {
entry:
  %call = tail call noalias ptr @_Znwm(i32 1)
  %call1 = tail call ptr @_ZN5AppleC1Ev(ptr %call)
  ret ptr %call
}

; CHECK-LABEL: _Z3barPvS_l:
; CHECK-NEXT: .functype _Z3barPvS_l (i32, i32, i32) -> (i32){{$}}
; CHECK-NEXT: {{^}} call     $push0=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
declare ptr @memcpy(ptr returned, ptr, i32)
define ptr @_Z3barPvS_l(ptr %p, ptr %s, i32 %n) {
entry:
  %call = tail call ptr @memcpy(ptr %p, ptr %s, i32 %n)
  ret ptr %p
}

; Test that the optimization isn't performed on constant arguments.

; CHECK-LABEL: test_constant_arg:
; CHECK:      i32.const   $push0=, global{{$}}
; CHECK-NEXT: {{^}} call        $drop=, returns_arg, $pop0{{$}}
; CHECK-NEXT: return{{$}}
@global = external global i32
@addr = global ptr @global
define void @test_constant_arg() {
  %call = call ptr @returns_arg(ptr @global)
  ret void
}
declare ptr @returns_arg(ptr returned)

; Test that the optimization isn't performed on arguments without the
; "returned" attribute.

; CHECK-LABEL: test_other_skipped:
; CHECK-NEXT: .functype test_other_skipped (i32, i32, f64) -> (){{$}}
; CHECK-NEXT: {{^}} call     $drop=, do_something, $0, $1, $2{{$}}
; CHECK-NEXT: {{^}} call     do_something_with_i32, $1{{$}}
; CHECK-NEXT: {{^}} call     do_something_with_double, $2{{$}}
declare i32 @do_something(i32 returned, i32, double)
declare void @do_something_with_i32(i32)
declare void @do_something_with_double(double)
define void @test_other_skipped(i32 %a, i32 %b, double %c) {
    %call = call i32 @do_something(i32 %a, i32 %b, double %c)
    call void @do_something_with_i32(i32 %b)
    call void @do_something_with_double(double %c)
    ret void
}

; Test that the optimization is performed on arguments other than the first.

; CHECK-LABEL: test_second_arg:
; CHECK-NEXT: .functype test_second_arg (i32, i32) -> (i32){{$}}
; CHECK-NEXT: {{^}} call     $push0=, do_something_else, $0, $1{{$}}
; CHECK-NEXT: return   $pop0{{$}}
declare i32 @do_something_else(i32, i32 returned)
define i32 @test_second_arg(i32 %a, i32 %b) {
    %call = call i32 @do_something_else(i32 %a, i32 %b)
    ret i32 %b
}
