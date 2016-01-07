; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that the "returned" attribute is optimized effectively.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: _Z3foov:
; CHECK-NEXT: .result   i32{{$}}
; CHECK-NEXT: i32.const $push0=, 1{{$}}
; CHECK-NEXT: {{^}} i32.call      $push1=, _Znwm, $pop0{{$}}
; CHECK-NEXT: {{^}} i32.call      $push2=, _ZN5AppleC1Ev, $pop1{{$}}
; CHECK-NEXT: return    $pop2{{$}}
%class.Apple = type { i8 }
declare noalias i8* @_Znwm(i32)
declare %class.Apple* @_ZN5AppleC1Ev(%class.Apple* returned)
define %class.Apple* @_Z3foov() {
entry:
  %call = tail call noalias i8* @_Znwm(i32 1)
  %0 = bitcast i8* %call to %class.Apple*
  %call1 = tail call %class.Apple* @_ZN5AppleC1Ev(%class.Apple* %0)
  ret %class.Apple* %0
}

; CHECK-LABEL: _Z3barPvS_l:
; CHECK-NEXT: .param   i32, i32, i32{{$}}
; CHECK-NEXT: .result  i32{{$}}
; CHECK-NEXT: {{^}} i32.call     $push0=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
declare i8* @memcpy(i8* returned, i8*, i32)
define i8* @_Z3barPvS_l(i8* %p, i8* %s, i32 %n) {
entry:
  %call = tail call i8* @memcpy(i8* %p, i8* %s, i32 %n)
  ret i8* %p
}

; Test that the optimization isn't performed on constant arguments.

; CHECK-LABEL: test_constant_arg:
; CHECK-NEXT: i32.const   $push0=, global{{$}}
; CHECK-NEXT: {{^}} i32.call        $discard=, returns_arg, $pop0{{$}}
; CHECK-NEXT: return{{$}}
@global = external global i32
@addr = global i32* @global
define void @test_constant_arg() {
  %call = call i32* @returns_arg(i32* @global)
  ret void
}
declare i32* @returns_arg(i32* returned)
