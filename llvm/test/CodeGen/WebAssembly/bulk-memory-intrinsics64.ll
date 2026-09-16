; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=+bulk-memory,+multimemory | FileCheck %s

target triple = "wasm64-unknown-unknown"

declare void @llvm.wasm.memory.copy.i64(i32, i32, ptr, ptr, i64)
declare void @llvm.wasm.memory.fill.i64(i32, ptr, i32, i64)

; CHECK-LABEL: memory_copy:
; CHECK-NEXT: .functype memory_copy (i64, i64, i64) -> ()
; CHECK-NEXT: memory.copy 0, 0, $0, $1, $2
; CHECK-NEXT: return
define void @memory_copy(ptr %dest, ptr %src, i64 %len) {
  call void @llvm.wasm.memory.copy.i64(i32 0, i32 0, ptr %dest, ptr %src, i64 %len)
  ret void
}

; CHECK-LABEL: memory_fill:
; CHECK-NEXT: .functype memory_fill (i64, i32, i64) -> ()
; CHECK-NEXT: memory.fill 0, $0, $1, $2
; CHECK-NEXT: return
define void @memory_fill(ptr %dest, i32 %value, i64 %len) {
  call void @llvm.wasm.memory.fill.i64(i32 0, ptr %dest, i32 %value, i64 %len)
  ret void
}

; CHECK-LABEL: memory_copy_multi:
; CHECK-NEXT: .functype memory_copy_multi (i64, i64, i64) -> ()
; CHECK-NEXT: memory.copy 1, 2, $0, $1, $2
; CHECK-NEXT: return
define void @memory_copy_multi(ptr %dest, ptr %src, i64 %len) {
  call void @llvm.wasm.memory.copy.i64(i32 1, i32 2, ptr %dest, ptr %src, i64 %len)
  ret void
}

; CHECK-LABEL: memory_fill_multi:
; CHECK-NEXT: .functype memory_fill_multi (i64, i32, i64) -> ()
; CHECK-NEXT: memory.fill 3, $0, $1, $2
; CHECK-NEXT: return
define void @memory_fill_multi(ptr %dest, i32 %value, i64 %len) {
  call void @llvm.wasm.memory.fill.i64(i32 3, ptr %dest, i32 %value, i64 %len)
  ret void
}
