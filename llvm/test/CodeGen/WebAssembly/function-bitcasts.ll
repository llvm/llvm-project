; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that function pointer casts are replaced with wrappers.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: test:
; CHECK-NEXT: call        .Lbitcast@FUNCTION{{$}}
; CHECK-NEXT: call        .Lbitcast@FUNCTION{{$}}
; CHECK-NEXT: call        .Lbitcast.1@FUNCTION{{$}}
; CHECK-NEXT: i32.const   $push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: call        .Lbitcast.2@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const   $push[[L1:[0-9]+]]=, 0
; CHECK-NEXT: call        .Lbitcast.2@FUNCTION, $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const   $push[[L2:[0-9]+]]=, 0
; CHECK-NEXT: call        .Lbitcast.2@FUNCTION, $pop[[L2]]{{$}}
; CHECK-NEXT: call        foo0@FUNCTION
; CHECK-NEXT: i32.call    $drop=, .Lbitcast.3@FUNCTION{{$}}
; CHECK-NEXT: call        foo2@FUNCTION{{$}}
; CHECK-NEXT: call        foo1@FUNCTION{{$}}
; CHECK-NEXT: call        foo3@FUNCTION{{$}}
; CHECK-NEXT: .endfunc

; CHECK-LABEL: .Lbitcast:
; CHECK-NEXT: .local      i32
; CHECK-NEXT: call        has_i32_arg@FUNCTION, $0{{$}}
; CHECK-NEXT: .endfunc

; CHECK-LABEL: .Lbitcast.1:
; CHECK-NEXT: call        $drop=, has_i32_ret@FUNCTION{{$}}
; CHECK-NEXT: .endfunc

; CHECK-LABEL: .Lbitcast.2:
; CHECK-NEXT: .param      i32
; CHECK-NEXT: call        foo0@FUNCTION{{$}}
; CHECK-NEXT: .endfunc

; CHECK-LABEL: .Lbitcast.3:
; CHECK-NEXT: .result     i32
; CHECK-NEXT: .local      i32
; CHECK-NEXT: call        foo1@FUNCTION{{$}}
; CHECK-NEXT: copy_local  $push0=, $0
; CHECK-NEXT: .endfunc

declare void @has_i32_arg(i32)
declare i32 @has_i32_ret()

declare void @foo0()
declare void @foo1()
declare void @foo2()
declare void @foo3()

define void @test() {
entry:
  call void bitcast (void (i32)* @has_i32_arg to void ()*)()
  call void bitcast (void (i32)* @has_i32_arg to void ()*)()
  call void bitcast (i32 ()* @has_i32_ret to void ()*)()
  call void bitcast (void ()* @foo0 to void (i32)*)(i32 0)
  %p = bitcast void ()* @foo0 to void (i32)*
  call void %p(i32 0)
  %q = bitcast void ()* @foo0 to void (i32)*
  call void %q(i32 0)
  %r = bitcast void (i32)* %q to void ()*
  call void %r()
  %t = call i32 bitcast (void ()* @foo1 to i32 ()*)()
  call void bitcast (void ()* @foo2 to void ()*)()
  call void @foo1()
  call void @foo3()

  ret void
}
