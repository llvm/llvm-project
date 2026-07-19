; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s

; Verify llvm.dx.interlocked.or expands to atomicrmw or monotonic.

; Groupshared (addrspace 3) memory tests.
@gs_i32 = internal addrspace(3) global i32 zeroinitializer
@gs_i64 = internal addrspace(3) global i64 zeroinitializer

define i32 @test_i32(i32 %v) {
entry:
; CHECK-LABEL: @test_i32
; CHECK: %[[R:.*]] = atomicrmw or ptr addrspace(3) @gs_i32, i32 %v monotonic
; CHECK: ret i32 %[[R]]
  %r = call i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3) @gs_i32, i32 %v)
  ret i32 %r
}

define i64 @test_i64(i64 %v) {
entry:
; CHECK-LABEL: @test_i64
; CHECK: %[[R:.*]] = atomicrmw or ptr addrspace(3) @gs_i64, i64 %v monotonic
; CHECK: ret i64 %[[R]]
  %r = call i64 @llvm.dx.interlocked.or.i64.p3(ptr addrspace(3) @gs_i64, i64 %v)
  ret i64 %r
}

; Device (addrspace 1) memory tests.
@dev_i32 = external addrspace(1) global i32
@dev_i64 = external addrspace(1) global i64

define i32 @test_device_i32(i32 %v) {
entry:
; CHECK-LABEL: @test_device_i32
; CHECK: %[[R:.*]] = atomicrmw or ptr addrspace(1) @dev_i32, i32 %v monotonic
; CHECK: ret i32 %[[R]]
  %r = call i32 @llvm.dx.interlocked.or.i32.p1(ptr addrspace(1) @dev_i32, i32 %v)
  ret i32 %r
}

define i64 @test_device_i64(i64 %v) {
entry:
; CHECK-LABEL: @test_device_i64
; CHECK: %[[R:.*]] = atomicrmw or ptr addrspace(1) @dev_i64, i64 %v monotonic
; CHECK: ret i64 %[[R]]
  %r = call i64 @llvm.dx.interlocked.or.i64.p1(ptr addrspace(1) @dev_i64, i64 %v)
  ret i64 %r
}

declare i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3), i32)
declare i64 @llvm.dx.interlocked.or.i64.p3(ptr addrspace(3), i64)
declare i32 @llvm.dx.interlocked.or.i32.p1(ptr addrspace(1), i32)
declare i64 @llvm.dx.interlocked.or.i64.p1(ptr addrspace(1), i64)
