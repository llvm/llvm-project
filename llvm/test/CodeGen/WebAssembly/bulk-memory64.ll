; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+bulk-memory | FileCheck %s --check-prefixes CHECK,BULK-MEM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=-bulk-memory | FileCheck %s --check-prefixes CHECK,NO-BULK-MEM

; Test that basic bulk memory codegen works correctly

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm64-unknown-unknown"

declare void @llvm.memcpy.p0i8.p0i8.i8(i8*, i8*, i8, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
declare void @llvm.memcpy.p0i32.p0i32.i64(i32*, i32*, i64, i1)

declare void @llvm.memmove.p0i8.p0i8.i8(i8*, i8*, i8, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i1)
declare void @llvm.memmove.p0i32.p0i32.i64(i32*, i32*, i64, i1)

declare void @llvm.memset.p0i8.i8(i8*, i8, i8, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)
declare void @llvm.memset.p0i32.i64(i32*, i8, i64, i1)

; CHECK-LABEL: memcpy_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i8 (i64, i64, i32) -> ()
; BULK-MEM-NEXT: i64.extend_i32_u $push0=, $2
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop0
; BULK-MEM-NEXT: return
define void @memcpy_i8(i8* %dest, i8* %src, i8 zeroext %len) {
  call void @llvm.memcpy.p0i8.p0i8.i8(i8* %dest, i8* %src, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i8 (i64, i64, i32) -> ()
; BULK-MEM-NEXT: i64.extend_i32_u $push0=, $2
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop0
; BULK-MEM-NEXT: return
define void @memmove_i8(i8* %dest, i8* %src, i8 zeroext %len) {
  call void @llvm.memmove.p0i8.p0i8.i8(i8* %dest, i8* %src, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memset_i8:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_i8 (i64, i32, i32) -> ()
; BULK-MEM-NEXT: i64.extend_i32_u $push0=, $2
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $pop0
; BULK-MEM-NEXT: return
define void @memset_i8(i8* %dest, i8 %val, i8 zeroext %len) {
  call void @llvm.memset.p0i8.i8(i8* %dest, i8 %val, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i32 (i64, i64, i64) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memcpy_i32(i32* %dest, i32* %src, i64 %len) {
  call void @llvm.memcpy.p0i32.p0i32.i64(i32* %dest, i32* %src, i64 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i32 (i64, i64, i64) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memmove_i32(i32* %dest, i32* %src, i64 %len) {
  call void @llvm.memmove.p0i32.p0i32.i64(i32* %dest, i32* %src, i64 %len, i1 0)
  ret void
}

; CHECK-LABEL: memset_i32:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_i32 (i64, i32, i64) -> ()
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memset_i32(i32* %dest, i8 %val, i64 %len) {
  call void @llvm.memset.p0i32.i64(i32* %dest, i8 %val, i64 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1:
; CHECK-NEXT: .functype memcpy_1 (i64, i64) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memcpy_1(i8* %dest, i8* %src) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1:
; CHECK-NEXT: .functype memmove_1 (i64, i64) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memmove_1(i8* %dest, i8* %src) {
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1, i1 0)
  ret void
}

; CHECK-LABEL: memset_1:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_1 (i64, i32) -> ()
; BULK-MEM-NEXT: i32.store8 0($0), $1
; BULK-MEM-NEXT: return
define void @memset_1(i8* %dest, i8 %val) {
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 1, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_1024 (i64, i64) -> ()
; BULK-MEM-NEXT: i64.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memcpy_1024(i8* %dest, i8* %src) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1024, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_1024 (i64, i64) -> ()
; BULK-MEM-NEXT: i64.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memmove_1024(i8* %dest, i8* %src) {
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1024, i1 0)
  ret void
}

; CHECK-LABEL: memset_1024:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_1024 (i64, i32) -> ()
; BULK-MEM-NEXT: i64.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memset_1024(i8* %dest, i8 %val) {
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 1024, i1 0)
  ret void
}

; The following tests check that frame index elimination works for
; bulk memory instructions. The stack pointer is bumped by 112 instead
; of 100 because the stack pointer in WebAssembly is currently always
; 16-byte aligned, even in leaf functions, although it is not written
; back to the global in this case.

; TODO: Change TransientStackAlignment to 1 to avoid this extra
; arithmetic. This will require forcing the use of StackAlignment in
; PrologEpilogEmitter.cpp when
; WebAssemblyFrameLowering::needsSPWriteback would be true.

; CHECK-LABEL: memcpy_alloca_src:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_alloca_src (i64) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i64.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i64.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i64.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i64.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i64.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $pop[[L4]], $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memcpy_alloca_src(i8* %dst) {
  %a = alloca [100 x i8]
  %p = bitcast [100 x i8]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %p, i64 100, i1 false)
  ret void
}

; CHECK-LABEL: memcpy_alloca_dst:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_alloca_dst (i64) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i64.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i64.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i64.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i64.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i64.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.copy 0, 0, $pop[[L4]], $0, $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memcpy_alloca_dst(i8* %src) {
  %a = alloca [100 x i8]
  %p = bitcast [100 x i8]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p, i8* %src, i64 100, i1 false)
  ret void
}

; CHECK-LABEL: memset_alloca:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_alloca (i32) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i64.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i64.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i64.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i64.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i64.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.fill 0, $pop[[L4]], $0, $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memset_alloca(i8 %val) {
  %a = alloca [100 x i8]
  %p = bitcast [100 x i8]* %a to i8*
  call void @llvm.memset.p0i8.i64(i8* %p, i8 %val, i64 100, i1 false)
  ret void
}
