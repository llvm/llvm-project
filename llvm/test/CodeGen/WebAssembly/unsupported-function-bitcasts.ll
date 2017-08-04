; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that function pointer casts that require conversions are not converted
; to wrappers. In theory some conversions could be supported, but currently no
; conversions are implemented.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: test:
; CHECK-NEXT: i32.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: call        has_i64_arg@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.call    $push{{[0-9]+}}=, has_i64_ret@FUNCTION{{$}}
; CHECK-NEXT: drop
; CHECK-NEXT: end_function

; CHECK-NOT: .Lbitcast

declare void @has_i64_arg(i64)
declare i64 @has_i64_ret()

define void @test() {
entry:
  call void bitcast (void (i64)* @has_i64_arg to void (i32)*)(i32 0)
  %t = call i32 bitcast (i64 ()* @has_i64_ret to i32 ()*)()
  ret void
}
