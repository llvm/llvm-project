; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S %s -o - | FileCheck %s
target triple = "thumbv7-apple-ios7.0.0"


%struct = type [1000 x i32]

define void @test_dead_gep(ptr %t0) {
; CHECK-LABEL: define void @test_dead_gep
; CHECK-NOT: getelementptr
; CHECK: %t16 = load i32, ptr undef
; CHECK: ret void

  %t12 = getelementptr inbounds %struct, ptr %t0, i32 1, i32 500
  %t13 = load i32, ptr %t12, align 4
  %t14 = icmp eq i32 %t13, 2
  %t15 = select i1 %t14, ptr undef, ptr undef
  %t16 = load i32, ptr %t15, align 4
  ret void
}
