; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+bulk-memory | FileCheck %s

; Test that bulk memory intrinsics lower correctly

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: memory_init:
; CHECK-NEXT: .functype  memory_init (i32, i32, i32) -> ()
; CHECK-NEXT: memory.init 3, 0, $0, $1, $2
; CHECK-NEXT: return
declare void @llvm.wasm.memory.init(i32, i32, i8*, i32, i32)
define void @memory_init(i8* %dest, i32 %offset, i32 %size) {
  call void @llvm.wasm.memory.init(
    i32 3, i32 0, i8* %dest, i32 %offset, i32 %size
  )
  ret void
}

; CHECK-LABEL: data_drop:
; CHECK-NEXT: .functype data_drop () -> ()
; CHECK-NEXT: data.drop 3
; CHECK-NEXT: return
declare void @llvm.wasm.data.drop(i32)
define void @data_drop() {
  call void @llvm.wasm.data.drop(i32 3)
  ret void
}
