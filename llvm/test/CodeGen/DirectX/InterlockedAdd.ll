; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.6-compute %s | FileCheck %s

; Verify llvm.dx.interlocked.add expands to atomicrmw add monotonic.

@gs_i32 = internal addrspace(3) global i32 zeroinitializer
@gs_i64 = internal addrspace(3) global i64 zeroinitializer

define i32 @test_i32(i32 %v) {
entry:
; CHECK-LABEL: @test_i32
; CHECK: %[[R:.*]] = atomicrmw add ptr addrspace(3) @gs_i32, i32 %v monotonic
; CHECK: ret i32 %[[R]]
  %r = call i32 @llvm.dx.interlocked.add.i32.p3(ptr addrspace(3) @gs_i32, i32 %v)
  ret i32 %r
}

define i64 @test_i64(i64 %v) {
entry:
; CHECK-LABEL: @test_i64
; CHECK: %[[R:.*]] = atomicrmw add ptr addrspace(3) @gs_i64, i64 %v monotonic
; CHECK: ret i64 %[[R]]
  %r = call i64 @llvm.dx.interlocked.add.i64.p3(ptr addrspace(3) @gs_i64, i64 %v)
  ret i64 %r
}

declare i32 @llvm.dx.interlocked.add.i32.p3(ptr addrspace(3), i32)
declare i64 @llvm.dx.interlocked.add.i64.p3(ptr addrspace(3), i64)
