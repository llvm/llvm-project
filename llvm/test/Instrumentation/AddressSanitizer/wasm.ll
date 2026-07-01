; Check ASan shadow mapping on WebAssembly (both wasm32 and wasm64 should use offset 0).
; RUN: opt -passes=asan -S -mtriple=wasm32-unknown-emscripten < %s | FileCheck %s --check-prefixes=WASM32
; RUN: opt -passes=asan -S -mtriple=wasm64-unknown-emscripten < %s | FileCheck %s --check-prefixes=WASM64

define i32 @test_load(ptr %a) sanitize_address {
; WASM32-LABEL: @test_load
; WASM32:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint ptr %a to i32
; WASM32:   %[[SHIFT:[^ ]*]] = lshr i32 %[[LOAD_ADDR]], 3
; WASM32-NOT: or i32 %[[SHIFT]]
; WASM32-NOT: add i32 %[[SHIFT]]
; WASM32:   inttoptr i32 %[[SHIFT]] to ptr

; WASM64-LABEL: @test_load
; WASM64:   %[[LOAD_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; WASM64:   %[[SHIFT:[^ ]*]] = lshr i64 %[[LOAD_ADDR]], 3
; WASM64-NOT: or i64 %[[SHIFT]]
; WASM64-NOT: add i64 %[[SHIFT]]
; WASM64:   inttoptr i64 %[[SHIFT]] to ptr

entry:
  %x = load i32, ptr %a, align 4
  ret i32 %x
}
