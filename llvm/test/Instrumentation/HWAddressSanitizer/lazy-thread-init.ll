; RUN: opt -S -hwasan < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @bar([16 x i32]* %p)

define void @foo() sanitize_hwaddress "hwasan-abi"="interceptor" {
  ; CHECK: [[LOAD:%[^ ]*]] = load i64, i64* @__hwasan_tls
  ; CHECK: [[ICMP:%[^ ]*]] = icmp eq i64 [[LOAD]], 0
  ; CHECK: br i1 [[ICMP]], label %[[INIT:[^,]*]], label %[[CONT:[^,]*]], !prof [[PROF:![0-9]+]]

  ; CHECK: [[INIT]]:
  ; CHECK: call void @__hwasan_thread_enter()
  ; CHECK: [[RELOAD:%[^ ]*]] = load i64, i64* @__hwasan_tls
  ; CHECK: br label %[[CONT]]

  ; CHECK: [[CONT]]:
  ; CHECK: phi i64 [ [[LOAD]], %0 ], [ [[RELOAD]], %[[INIT]] ]

  %p = alloca [16 x i32]
  call void @bar([16 x i32]* %p)
  ret void
}

; CHECK: [[PROF]] = !{!"branch_weights", i32 1, i32 100000}
