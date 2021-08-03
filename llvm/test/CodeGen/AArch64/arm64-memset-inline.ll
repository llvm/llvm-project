; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define void @bzero_4_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_4_heap:
; CHECK:       str wzr, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 4 %c, i8 0, i64 4, i1 false)
  ret void
}

define void @bzero_8_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_8_heap:
; CHECK:       str xzr, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 8, i1 false)
  ret void
}

define void @bzero_12_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_12_heap:
; CHECK:       str wzr, [x0, #8]
; CHECK-NEXT:  str xzr, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 12, i1 false)
  ret void
}

define void @bzero_16_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_16_heap:
; CHECK:       stp xzr, xzr, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 16, i1 false)
  ret void
}

define void @bzero_32_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_32_heap:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  stp q0, q0, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 32, i1 false)
  ret void
}

define void @bzero_64_heap(i8* nocapture %c) {
; CHECK-LABEL: bzero_64_heap:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  stp q0, q0, [x0, #32]
; CHECK-NEXT:  stp q0, q0, [x0]
; CHECK-NEXT:  ret
  call void @llvm.memset.p0i8.i64(i8* align 8 %c, i8 0, i64 64, i1 false)
  ret void
}

define void @bzero_4_stack() {
; CHECK-LABEL: bzero_4_stack:
; CHECK:       str wzr, [sp, #12]
; CHECK-NEXT:  bl something
  %buf = alloca [4 x i8], align 1
  %cast = bitcast [4 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 4, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_8_stack() {
; CHECK-LABEL: bzero_8_stack:
; CHECK:       stp x30, xzr, [sp, #-16]!
; CHECK:       bl something
  %buf = alloca [8 x i8], align 1
  %cast = bitcast [8 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 8, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_12_stack() {
; CHECK-LABEL: bzero_12_stack:
; CHECK:       str wzr, [sp, #8]
; CHECK-NEXT:  str xzr, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [12 x i8], align 1
  %cast = bitcast [12 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 12, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_16_stack() {
; CHECK-LABEL: bzero_16_stack:
; CHECK:       stp xzr, x30, [sp, #8]
; CHECK:       mov x0, sp
; CHECK:       str xzr, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [16 x i8], align 1
  %cast = bitcast [16 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 16, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_20_stack() {
; CHECK-LABEL: bzero_20_stack:
; CHECK:       stp xzr, xzr, [sp, #8]
; CHECK-NEXT:  str wzr, [sp, #24]
; CHECK-NEXT:  bl something
  %buf = alloca [20 x i8], align 1
  %cast = bitcast [20 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 20, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_26_stack() {
; CHECK-LABEL: bzero_26_stack:
; CHECK:       stp xzr, xzr, [sp]
; CHECK-NEXT:  strh wzr, [sp, #24]
; CHECK-NEXT:  str xzr, [sp, #16]
; CHECK-NEXT:  bl something
  %buf = alloca [26 x i8], align 1
  %cast = bitcast [26 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 26, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_32_stack() {
; CHECK-LABEL: bzero_32_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [32 x i8], align 1
  %cast = bitcast [32 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 32, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_40_stack() {
; CHECK-LABEL: bzero_40_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  str xzr, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT: bl something
  %buf = alloca [40 x i8], align 1
  %cast = bitcast [40 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 40, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_64_stack() {
; CHECK-LABEL: bzero_64_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [64 x i8], align 1
  %cast = bitcast [64 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 64, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_72_stack() {
; CHECK-LABEL: bzero_72_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  str xzr, [sp, #64]
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [72 x i8], align 1
  %cast = bitcast [72 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 72, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_128_stack() {
; CHECK-LABEL: bzero_128_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp, #96]
; CHECK-NEXT:  stp q0, q0, [sp, #64]
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [128 x i8], align 1
  %cast = bitcast [128 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 128, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @bzero_256_stack() {
; CHECK-LABEL: bzero_256_stack:
; CHECK:       movi v0.2d, #0000000000000000
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp, #224]
; CHECK-NEXT:  stp q0, q0, [sp, #192]
; CHECK-NEXT:  stp q0, q0, [sp, #160]
; CHECK-NEXT:  stp q0, q0, [sp, #128]
; CHECK-NEXT:  stp q0, q0, [sp, #96]
; CHECK-NEXT:  stp q0, q0, [sp, #64]
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [256 x i8], align 1
  %cast = bitcast [256 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 0, i32 256, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_4_stack() {
; CHECK-LABEL: memset_4_stack:
; CHECK:       mov w8, #-1431655766
; CHECK-NEXT:  add x0, sp, #12
; CHECK-NEXT:  str w8, [sp, #12]
; CHECK-NEXT:  bl something
  %buf = alloca [4 x i8], align 1
  %cast = bitcast [4 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 4, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_8_stack() {
; CHECK-LABEL: memset_8_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  stp x30, x8, [sp, #-16]!
; CHECK-NEXT:  add x0, sp, #8
; CHECK-NEXT:  bl something
  %buf = alloca [8 x i8], align 1
  %cast = bitcast [8 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 8, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_12_stack() {
; CHECK-LABEL: memset_12_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  str x8, [sp]
; CHECK-NEXT:  str w8, [sp, #8]
; CHECK-NEXT:  bl something
  %buf = alloca [12 x i8], align 1
  %cast = bitcast [12 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 12, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_16_stack() {
; CHECK-LABEL: memset_16_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp x8, x30, [sp, #8]
; CHECK-NEXT:  str x8, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [16 x i8], align 1
  %cast = bitcast [16 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 16, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_20_stack() {
; CHECK-LABEL: memset_20_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  add x0, sp, #8
; CHECK-NEXT:  stp x8, x8, [sp, #8]
; CHECK-NEXT:  str w8, [sp, #24]
; CHECK-NEXT:  bl something
  %buf = alloca [20 x i8], align 1
  %cast = bitcast [20 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 20, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_26_stack() {
; CHECK-LABEL: memset_26_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp x8, x8, [sp, #8]
; CHECK-NEXT:  str x8, [sp]
; CHECK-NEXT:  strh w8, [sp, #24]
; CHECK-NEXT:  bl something
  %buf = alloca [26 x i8], align 1
  %cast = bitcast [26 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 26, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_32_stack() {
; CHECK-LABEL: memset_32_stack:
; CHECK:       movi v0.16b, #170
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [32 x i8], align 1
  %cast = bitcast [32 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 32, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_40_stack() {
; CHECK-LABEL: memset_40_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  movi v0.16b, #170
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  str x8, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT: bl something
  %buf = alloca [40 x i8], align 1
  %cast = bitcast [40 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 40, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_64_stack() {
; CHECK-LABEL: memset_64_stack:
; CHECK:       movi v0.16b, #170
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [64 x i8], align 1
  %cast = bitcast [64 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 64, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_72_stack() {
; CHECK-LABEL: memset_72_stack:
; CHECK:       mov x8, #-6148914691236517206
; CHECK-NEXT:  movi v0.16b, #170
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  str x8, [sp, #64]
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [72 x i8], align 1
  %cast = bitcast [72 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 72, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_128_stack() {
; CHECK-LABEL: memset_128_stack:
; CHECK:       movi v0.16b, #170
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  stp q0, q0, [sp, #96]
; CHECK-NEXT:  stp q0, q0, [sp, #64]
; CHECK-NEXT:  stp q0, q0, [sp, #32]
; CHECK-NEXT:  stp q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [128 x i8], align 1
  %cast = bitcast [128 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 128, i1 false)
  call void @something(i8* %cast)
  ret void
}

define void @memset_256_stack() {
; CHECK-LABEL: memset_256_stack:
; CHECK:       movi	v0.16b, #170
; CHECK-NEXT:  mov	x0, sp
; CHECK-NEXT:  stp	q0, q0, [sp, #224]
; CHECK-NEXT:  stp	q0, q0, [sp, #192]
; CHECK-NEXT:  stp	q0, q0, [sp, #160]
; CHECK-NEXT:  stp	q0, q0, [sp, #128]
; CHECK-NEXT:  stp	q0, q0, [sp, #96]
; CHECK-NEXT:  stp	q0, q0, [sp, #64]
; CHECK-NEXT:  stp	q0, q0, [sp, #32]
; CHECK-NEXT:  stp	q0, q0, [sp]
; CHECK-NEXT:  bl something
  %buf = alloca [256 x i8], align 1
  %cast = bitcast [256 x i8]* %buf to i8*
  call void @llvm.memset.p0i8.i32(i8* %cast, i8 -86, i32 256, i1 false)
  call void @something(i8* %cast)
  ret void
}

declare void @something(i8*)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
