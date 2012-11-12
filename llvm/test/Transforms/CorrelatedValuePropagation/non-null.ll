; RUN: opt < %s -correlated-propagation -S | FileCheck %s

define void @test1(i8* %ptr) {
; CHECK: test1
  %A = load i8* %ptr
  br label %bb
bb:
  icmp ne i8* %ptr, null
; CHECK-NOT: icmp
  ret void
}

define void @test2(i8* %ptr) {
; CHECK: test2
  store i8 0, i8* %ptr
  br label %bb
bb:
  icmp ne i8* %ptr, null
; CHECK-NOT: icmp
  ret void
}

define void @test3() {
; CHECK: test3
  %ptr = alloca i8
  br label %bb
bb:
  icmp ne i8* %ptr, null
; CHECK-NOT: icmp
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)
define void @test4(i8* %dest, i8* %src) {
; CHECK: test4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i32 1, i1 false)
  br label %bb
bb:
  icmp ne i8* %dest, null
  icmp ne i8* %src, null
; CHECK-NOT: icmp
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)
define void @test5(i8* %dest, i8* %src) {
; CHECK: test5
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i32 1, i1 false)
  br label %bb
bb:
  icmp ne i8* %dest, null
  icmp ne i8* %src, null
; CHECK-NOT: icmp
  ret void
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i32, i1)
define void @test6(i8* %dest) {
; CHECK: test6
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 255, i32 1, i32 1, i1 false)
  br label %bb
bb:
  icmp ne i8* %dest, null
; CHECK-NOT: icmp
  ret void
}

define void @test7(i8* %dest, i8* %src, i32 %len) {
; CHECK: test7
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i32 1, i1 false)
  br label %bb
bb:
  %KEEP1 = icmp ne i8* %dest, null
; CHECK: KEEP1
  %KEEP2 = icmp ne i8* %src, null
; CHECK: KEEP2
  ret void
}

declare void @llvm.memcpy.p1i8.p1i8.i32(i8 addrspace(1) *, i8 addrspace(1) *, i32, i32, i1)
define void @test8(i8 addrspace(1) * %dest, i8 addrspace(1) * %src) {
; CHECK: test8
  call void @llvm.memcpy.p1i8.p1i8.i32(i8 addrspace(1) * %dest, i8 addrspace(1) * %src, i32 1, i32 1, i1 false)
  br label %bb
bb:
  %KEEP1 = icmp ne i8 addrspace(1) * %dest, null
; CHECK: KEEP1
  %KEEP2 = icmp ne i8 addrspace(1) * %src, null
; CHECK: KEEP2
  ret void
}

define void @test9(i8* %dest, i8* %src) {
; CHECK: test9
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i32 1, i1 true)
  br label %bb
bb:
  %KEEP1 = icmp ne i8* %dest, null
; CHECK: KEEP1
  %KEEP2 = icmp ne i8* %src, null
; CHECK: KEEP2
  ret void
}
