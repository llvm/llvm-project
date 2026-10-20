; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The threadgroup index counts four-element vectors (base + 4 * element size
; * index): constant offsets that are multiples of that go in the immediate.

@tg = internal addrspace(3) global [256 x i32] undef

declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; CHECK-LABEL: offsets:
; CHECK-DAG:     threadgroup_load i32, x, {{r[0-9]+}}, [[B:r[0-9]+l]], 0
; CHECK-DAG:     threadgroup_load i32, x, {{r[0-9]+}}, [[B]], 1
; CHECK-DAG:     threadgroup_load i32, x, {{r[0-9]+}}, [[B]], 4
; CHECK-DAG:     threadgroup_load i16, x, {{r[0-9]+}}, [[B]], 3
; CHECK-DAG:     threadgroup_store i32, x, {{r[0-9]+}}, [[B]], 2
; CHECK:         stop
define agx_kernel void @offsets(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %b = getelementptr [256 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  %p16 = getelementptr i8, ptr addrspace(3) %b, i32 16
  %p64 = getelementptr i8, ptr addrspace(3) %b, i32 64
  %p24 = getelementptr i8, ptr addrspace(3) %b, i32 24
  %p32 = getelementptr i8, ptr addrspace(3) %b, i32 32
  %x = load i32, ptr addrspace(3) %b
  %y = load i32, ptr addrspace(3) %p16
  %z = load i32, ptr addrspace(3) %p64
  %h = load i16, ptr addrspace(3) %p24
  %hz = zext i16 %h to i32
  %s1 = add i32 %x, %y
  %s2 = add i32 %s1, %z
  %s3 = add i32 %s2, %hz
  store i32 %s3, ptr addrspace(3) %p32
  store i32 %s3, ptr addrspace(1) %out
  ret void
}

; An offset that is not a multiple of the vector stays in the base register.
; CHECK-LABEL: unaligned:
; CHECK:         threadgroup_load i32, x, {{r[0-9]+}}, {{r[0-9]+l}}, 0
define agx_kernel void @unaligned(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %b = getelementptr [256 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  %p = getelementptr i8, ptr addrspace(3) %b, i32 4
  %x = load i32, ptr addrspace(3) %p
  store i32 %x, ptr addrspace(1) %out
  ret void
}
