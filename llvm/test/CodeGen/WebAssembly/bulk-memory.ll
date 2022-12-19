; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=+bulk-memory | FileCheck %s --check-prefixes CHECK,BULK-MEM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mcpu=mvp -mattr=-bulk-memory | FileCheck %s --check-prefixes CHECK,NO-BULK-MEM

; Test that basic bulk memory codegen works correctly

target triple = "wasm32-unknown-unknown"

declare void @llvm.memcpy.p0.p0.i8(ptr, ptr, i8, i1)
declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)

declare void @llvm.memmove.p0.p0.i8(ptr, ptr, i8, i1)
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)

declare void @llvm.memset.p0.i8(ptr, i8, i8, i1)
declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)

; CHECK-LABEL: memcpy_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i8 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memcpy_i8(ptr %dest, ptr %src, i8 zeroext %len) {
  call void @llvm.memcpy.p0.p0.i8(ptr %dest, ptr %src, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i8:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i8 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memmove_i8(ptr %dest, ptr %src, i8 zeroext %len) {
  call void @llvm.memmove.p0.p0.i8(ptr %dest, ptr %src, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memset_i8:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_i8 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memset_i8(ptr %dest, i8 %val, i8 zeroext %len) {
  call void @llvm.memset.p0.i8(ptr %dest, i8 %val, i8 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_i32 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memcpy_i32(ptr %dest, ptr %src, i32 %len) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memmove_i32:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_i32 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memmove_i32(ptr %dest, ptr %src, i32 %len) {
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memset_i32:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_i32 (i32, i32, i32) -> ()
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $2
; BULK-MEM-NEXT: return
define void @memset_i32(ptr %dest, i8 %val, i32 %len) {
  call void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 %len, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1:
; CHECK-NEXT: .functype memcpy_1 (i32, i32) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memcpy_1(ptr %dest, ptr %src) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 1, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1:
; CHECK-NEXT: .functype memmove_1 (i32, i32) -> ()
; CHECK-NEXT: i32.load8_u $push[[L0:[0-9]+]]=, 0($1)
; CHECK-NEXT: i32.store8 0($0), $pop[[L0]]
; CHECK-NEXT: return
define void @memmove_1(ptr %dest, ptr %src) {
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 1, i1 0)
  ret void
}

; CHECK-LABEL: memset_1:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_1 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.store8 0($0), $1
; BULK-MEM-NEXT: return
define void @memset_1(ptr %dest, i8 %val) {
  call void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 1, i1 0)
  ret void
}

; CHECK-LABEL: memcpy_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_1024 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memcpy_1024(ptr %dest, ptr %src) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 1024, i1 0)
  ret void
}

; CHECK-LABEL: memmove_1024:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memmove_1024 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memmove_1024(ptr %dest, ptr %src) {
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 1024, i1 0)
  ret void
}

; CHECK-LABEL: memset_1024:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_1024 (i32, i32) -> ()
; BULK-MEM-NEXT: i32.const $push[[L0:[0-9]+]]=, 1024
; BULK-MEM-NEXT: memory.fill 0, $0, $1, $pop[[L0]]
; BULK-MEM-NEXT: return
define void @memset_1024(ptr %dest, i8 %val) {
  call void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 1024, i1 0)
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
; BULK-MEM-NEXT: .functype memcpy_alloca_src (i32) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i32.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i32.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i32.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i32.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.copy 0, 0, $0, $pop[[L4]], $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memcpy_alloca_src(ptr %dst) {
  %a = alloca [100 x i8]
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %a, i32 100, i1 false)
  ret void
}

; CHECK-LABEL: memcpy_alloca_dst:
; NO-BULK-MEM-NOT: memory.copy
; BULK-MEM-NEXT: .functype memcpy_alloca_dst (i32) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i32.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i32.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i32.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i32.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.copy 0, 0, $pop[[L4]], $0, $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memcpy_alloca_dst(ptr %src) {
  %a = alloca [100 x i8]
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %src, i32 100, i1 false)
  ret void
}

; CHECK-LABEL: memset_alloca:
; NO-BULK-MEM-NOT: memory.fill
; BULK-MEM-NEXT: .functype memset_alloca (i32) -> ()
; BULK-MEM-NEXT: global.get $push[[L0:[0-9]+]]=, __stack_pointer
; BULK-MEM-NEXT: i32.const $push[[L1:[0-9]+]]=, 112
; BULK-MEM-NEXT: i32.sub $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; BULK-MEM-NEXT: i32.const $push[[L3:[0-9]+]]=, 12
; BULK-MEM-NEXT: i32.add $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]
; BULK-MEM-NEXT: i32.const $push[[L5:[0-9]+]]=, 100
; BULK-MEM-NEXT: memory.fill 0, $pop[[L4]], $0, $pop[[L5]]
; BULK-MEM-NEXT: return
define void @memset_alloca(i8 %val) {
  %a = alloca [100 x i8]
  call void @llvm.memset.p0.i32(ptr %a, i8 %val, i32 100, i1 false)
  ret void
}
