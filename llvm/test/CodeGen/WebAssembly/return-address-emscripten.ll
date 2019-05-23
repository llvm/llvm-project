; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

; This tests the implementation of __builtin_return_address on emscripten

; CHECK-LABEL: test_returnaddress:
; CHECK-NEXT: .functype test_returnaddress () -> (i32){{$}}
; CHECK-NEXT: {{^}} i32.const 0{{$}}
; CHECK-NEXT: {{^}} i32.call emscripten_return_address{{$}}
; CHECK-NEXT: {{^}} end_function{{$}}
define i8* @test_returnaddress() {
  %r = call i8* @llvm.returnaddress(i32 0)
  ret i8* %r
}

; LLVM represents __builtin_return_address as call to this function in IR.
declare i8* @llvm.returnaddress(i32 immarg)
