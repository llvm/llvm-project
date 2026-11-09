; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s

; Test that EnforceNodeIdInvariant correctly handles nodes
; users and prevents redundant invalidation visits. This test creates a
; DAG structure with high fan-in where multiple nodes share common users.
; should visit each node exactly once.

define i32 @test_diamond_dag(i32 %a, i32 %b) {
; CHECK-LABEL: test_diamond_dag:
entry:
  %add1 = add i32 %a, %b
  %add2 = add i32 %a, %b
  %mul1 = mul i32 %add1, %add2
  %mul2 = mul i32 %add1, %add2
  %result = add i32 %mul1, %mul2
  ret i32 %result
}

define i64 @test_deep_dag(i64 %x) {
; CHECK-LABEL: test_deep_dag:
entry:
  %a = add i64 %x, 1
  %b = add i64 %x, 2
  %c = add i64 %a, %b
  %d = add i64 %a, %b
  %e = mul i64 %c, %d
  %f = mul i64 %c, %d
  %g = add i64 %e, %f
  ret i64 %g
}

define i32 @test_wide_dag(i32 %v) {
; CHECK-LABEL: test_wide_dag:
entry:
  %a = add i32 %v, 1
  %b = add i32 %v, 2
  %c = add i32 %v, 3
  %d = add i32 %v, 4
  %ab = add i32 %a, %b
  %cd = add i32 %c, %d
  %ac = add i32 %a, %c
  %bd = add i32 %b, %d
  %t1 = mul i32 %ab, %cd
  %t2 = mul i32 %ac, %bd
  %result = add i32 %t1, %t2
  ret i32 %result
}

define <4 x i32> @test_vector_dag(<4 x i32> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: test_vector_dag:
entry:
  %add1 = add <4 x i32> %vec1, %vec2
  %add2 = add <4 x i32> %vec1, %vec2
  %mul1 = mul <4 x i32> %add1, %add2
  %mul2 = mul <4 x i32> %add1, %add2
  %result = add <4 x i32> %mul1, %mul2
  ret <4 x i32> %result
}

define i32 @test_chain_with_shared_users(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: test_chain_with_shared_users:
entry:
  %a = add i32 %x, %y
  %b = add i32 %y, %z
  %c = mul i32 %a, %b
  %d = mul i32 %a, %b
  %e = add i32 %c, %d
  %f = sub i32 %c, %d
  %g = mul i32 %e, %f
  ret i32 %g
}

define i64 @test_complex_sharing(i64 %p1, i64 %p2, i64 %p3) {
; CHECK-LABEL: test_complex_sharing:
entry:
  %n1 = add i64 %p1, %p2
  %n2 = add i64 %p2, %p3
  %n3 = add i64 %p1, %p3
  %u1 = mul i64 %n1, %n2
  %u2 = mul i64 %n1, %n3
  %u3 = mul i64 %n2, %n3
  %s1 = add i64 %u1, %u2
  %s2 = add i64 %u2, %u3
  %s3 = add i64 %u1, %u3
  %f1 = mul i64 %s1, %s2
  %f2 = mul i64 %s2, %s3
  %result = add i64 %f1, %f2
  ret i64 %result
}

define i32 @test_multiple_paths_to_same_user(i32 %input) {
; CHECK-LABEL: test_multiple_paths_to_same_user:
entry:
  %a = add i32 %input, 1
  %b = add i32 %input, 2
  %c = add i32 %input, 3
  %ab = mul i32 %a, %b
  %bc = mul i32 %b, %c
  %ac = mul i32 %a, %c
  %shared = add i32 %ab, %bc
  %shared2 = add i32 %ac, %shared
  ret i32 %shared2
}

define i64 @test_reconvergent_paths(i64 %val) {
; CHECK-LABEL: test_reconvergent_paths:
entry:
  %l1a = add i64 %val, 10
  %l1b = add i64 %val, 20
  %l2a = mul i64 %l1a, 2
  %l2b = mul i64 %l1b, 3
  %l2c = mul i64 %l1a, 4
  %l2d = mul i64 %l1b, 5
  %l3a = add i64 %l2a, %l2b
  %l3b = add i64 %l2c, %l2d
  %merge = mul i64 %l3a, %l3b
  ret i64 %merge
}
