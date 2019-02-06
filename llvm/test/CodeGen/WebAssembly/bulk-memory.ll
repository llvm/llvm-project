; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+bulk-memory | FileCheck %s --check-prefixes CHECK,BULK-MEM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=-bulk-memory | FileCheck %s --check-prefixes CHECK,NO-BULK-MEM

; Test that basic bulk memory codegen works correctly

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: memcpy_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i8 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy $0, $1, $2
; BULK-MEM-NEXT: return
declare void @llvm.memcpy.p0i8.p0i8.i32(
  i8* %dest, i8* %src, i32 %len, i1 %volatile
)
define void @memcpy_i8(i8* %dest, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i8 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy $0, $1, $2
; BULK-MEM-NEXT: return
declare void @llvm.memmove.p0i8.p0i8.i32(
  i8* %dest, i8* %src, i32 %len, i1 %volatile
)
define void @memmove_i8(i8* %dest, i8* %src, i32 %len) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i32 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy $0, $1, $2
; BULK-MEM-NEXT: return
declare void @llvm.memcpy.p0i32.p0i32.i32(
  i32* %dest, i32* %src, i32 %len, i1 %volatile
)
define void @memcpy_i32(i32* %dest, i32* %src, i32 %len) {
  call void @llvm.memcpy.p0i32.p0i32.i32(i32* %dest, i32* %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i32 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy $0, $1, $2
; BULK-MEM-NEXT: return
declare void @llvm.memmove.p0i32.p0i32.i32(
  i32* %dest, i32* %src, i32 %len, i1 %volatile
)
define void @memmove_i32(i32* %dest, i32* %src, i32 %len) {
  call void @llvm.memmove.p0i32.p0i32.i32(i32* %dest, i32* %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1:
; CHECK-NEXT: .functype memcpy_1 (i32, i32) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memcpy_1(i8* %dest, i8* %src) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1:
; CHECK-NEXT: .functype memmove_1 (i32, i32) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memmove_1(i8* %dest, i8* %src) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_1024 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memcpy_1024(i8* %dest, i8* %src) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1024, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_1024 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memmove_1024(i8* %dest, i8* %src) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1024, i1 0)
  ret void
}
