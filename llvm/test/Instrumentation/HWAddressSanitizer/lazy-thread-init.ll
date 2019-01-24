; RUN: opt -S -hwasan -hwasan-allow-ifunc < %s | FileCheck %s

target triple = "aarch64--linux-android"

declare void @bar([16 x i32]* %p)

define void @alloca() sanitize_hwaddress "hwasan-abi"="interceptor" {
  ; CHECK: [[A:%[^ ]*]] = call i8* @llvm.thread.pointer()
  ; CHECK: [[B:%[^ ]*]] = getelementptr i8, i8* [[A]], i32 48
  ; CHECK: [[C:%[^ ]*]] = bitcast i8* [[B]] to i64*
  ; CHECK: [[LOAD:%[^ ]*]] = load i64, i64* [[C]]
  ; CHECK: [[ICMP:%[^ ]*]] = icmp eq i64 [[LOAD]], 0
  ; CHECK: br i1 [[ICMP]], label %[[INIT:[^,]*]], label %[[CONT:[^,]*]], !prof [[PROF:![0-9]+]]

  ; CHECK: [[INIT]]:
  ; CHECK: call void @__hwasan_thread_enter()
  ; CHECK: [[RELOAD:%[^ ]*]] = load i64, i64* [[C]]
  ; CHECK: br label %[[CONT]]

  ; CHECK: [[CONT]]:
  ; CHECK: phi i64 [ [[LOAD]], %0 ], [ [[RELOAD]], %[[INIT]] ]

  %p = alloca [16 x i32]
  call void @bar([16 x i32]* %p)
  ret void
}

define i32 @load(i32* %p) sanitize_hwaddress "hwasan-abi"="interceptor" {
  ; CHECK: [[SHADOW:%[^ ]*]] = call i8* asm "", "=r,0"([0 x i8]* @__hwasan_shadow)
  ; CHECK-NOT: icmp
  ; CHECK: call void @llvm.hwasan.check.memaccess(i8* [[SHADOW]],
  %v = load i32, i32* %p
  ret i32 %v
}

; CHECK: [[PROF]] = !{!"branch_weights", i32 1, i32 100000}
