; RUN: not llc < %s -asm-verbose=false 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; This tests the implementation of __builtin_return_address on the unknown OS.
; Since this is not implemented, it should fail.

; CHECK: Non-Emscripten WebAssembly hasn't implemented __builtin_return_address
define i8* @test_returnaddress() {
  %r = call i8* @llvm.returnaddress(i32 0)
  ret i8* %r
}

; LLVM represents __builtin_return_address as call to this function in IR.
declare i8* @llvm.returnaddress(i32 immarg)
