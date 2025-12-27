; Test basic NoAlias sanitizer instrumentation.
;
; RUN: opt < %s -passes=nasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Test that noalias parameters are instrumented with provenance creation/destruction
define void @test_noalias_param(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: @test_noalias_param
; CHECK: %[[PA:.*]] = call i64 @__nasan_create_provenance(ptr %a
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[PA]])
; CHECK: %[[PB:.*]] = call i64 @__nasan_create_provenance(ptr %b
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %b, i64 %[[PB]])
; CHECK: %nasan.merge.array = alloca ptr
; CHECK: call void @__nasan_destroy_provenance(i64 %[[PB]])
; CHECK: call void @__nasan_destroy_provenance(i64 %[[PA]])
; CHECK: ret void
entry:
  ret void
}

; Test that loads through noalias pointers are checked
define i32 @test_load(ptr noalias %a) {
; CHECK-LABEL: @test_load
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: %nasan.merge.array = alloca ptr
; CHECK: call void @__nasan_check_load(i64 %{{.*}}, i64 4, i64 %[[P]])
; CHECK-NEXT: %tmp1 = load i32, ptr %a
; CHECK: call void @__nasan_destroy_provenance(i64 %[[P]])
; CHECK: ret i32 %tmp1
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

; Test that stores through noalias pointers are checked
define void @test_store(ptr noalias %a) {
; CHECK-LABEL: @test_store
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: %nasan.merge.array = alloca ptr
; CHECK: call void @__nasan_check_store(i64 %{{.*}}, i64 4, i64 %[[P]])
; CHECK-NEXT: store i32 42, ptr %a
; CHECK: call void @__nasan_destroy_provenance(i64 %[[P]])
; CHECK: ret void
entry:
  store i32 42, ptr %a, align 4
  ret void
}

; Test that GEP inherits provenance from base pointer
define void @test_gep(ptr noalias %a) {
; CHECK-LABEL: @test_gep
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: %gep = getelementptr i32, ptr %a, i64 1
; CHECK-NEXT: call void @__nasan_inherit_provenance(ptr %gep, ptr %a)
; CHECK: call void @__nasan_check_store(i64 %{{.*}}, i64 4, i64 %[[P]])
; CHECK-NEXT: store i32 42, ptr %gep
entry:
  %gep = getelementptr i32, ptr %a, i64 1
  store i32 42, ptr %gep, align 4
  ret void
}

; Test that functions without noalias params are not instrumented
define i32 @test_no_noalias(ptr %a) {
; CHECK-LABEL: @test_no_noalias
; CHECK-NOT: call {{.*}} @__nasan
; CHECK: %tmp1 = load i32, ptr %a
; CHECK: ret i32 %tmp1
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

; Test pointer load provenance propagation
define ptr @test_pointer_load(ptr noalias %a) {
; CHECK-LABEL: @test_pointer_load
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: call void @__nasan_check_load
; CHECK-NEXT: %ptr = load ptr, ptr %a
; CHECK-NEXT: call void @__nasan_propagate_through_load(ptr %ptr, ptr %a)
entry:
  %ptr = load ptr, ptr %a, align 8
  ret ptr %ptr
}

; Test pointer store provenance recording
define void @test_pointer_store(ptr noalias %a, ptr %val) {
; CHECK-LABEL: @test_pointer_store
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: call void @__nasan_check_store
; CHECK: call i64 @__nasan_get_pointer_provenance(ptr %val)
; CHECK-NEXT: call void @__nasan_record_pointer_store(ptr %a, i64
; CHECK-NEXT: store ptr %val, ptr %a
entry:
  store ptr %val, ptr %a, align 8
  ret void
}

; Test PHI node provenance merging
define void @test_phi(ptr noalias %a, ptr noalias %b, i1 %cond) {
; CHECK-LABEL: @test_phi
; CHECK: call i64 @__nasan_create_provenance(ptr %a
; CHECK: call i64 @__nasan_create_provenance(ptr %b
entry:
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
; CHECK: %ptr = phi ptr [ %a, %then ], [ %b, %else ]
; CHECK: call void @__nasan_merge_provenance(ptr %ptr, ptr %nasan.merge.array, i64 2)
  %ptr = phi ptr [ %a, %then ], [ %b, %else ]
  store i32 42, ptr %ptr
  ret void
}

; Test select instruction provenance merging
define void @test_select(ptr noalias %a, ptr noalias %b, i1 %cond) {
; CHECK-LABEL: @test_select
; CHECK: call i64 @__nasan_create_provenance(ptr %a
; CHECK: call i64 @__nasan_create_provenance(ptr %b
; CHECK: %ptr = select i1 %cond, ptr %a, ptr %b
; CHECK: call void @__nasan_merge_provenance(ptr %ptr, ptr %nasan.merge.array, i64 2)
entry:
  %ptr = select i1 %cond, ptr %a, ptr %b
  store i32 42, ptr %ptr
  ret void
}

; Test inttoptr creates zero provenance (unknown provenance)
define void @test_inttoptr(ptr noalias %a, i64 %addr) {
; CHECK-LABEL: @test_inttoptr
; CHECK: %[[P:.*]] = call i64 @__nasan_create_provenance
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %a, i64 %[[P]])
; CHECK: %ptr = inttoptr i64 %addr to ptr
; CHECK-NEXT: call void @__nasan_set_pointer_provenance(ptr %ptr, i64 0)
entry:
  %ptr = inttoptr i64 %addr to ptr
  store i32 42, ptr %ptr
  ret void
}

; Test that non-escaping allocas are not instrumented (optimization)
define void @test_local_alloca(ptr noalias %a) {
; CHECK-LABEL: @test_local_alloca
; CHECK: %local = alloca i32
; Non-escaping allocas should NOT be checked
; CHECK-NOT: call void @__nasan_check_store({{.*}}%local
; CHECK: store i32 42, ptr %local
; But accesses through noalias param should be checked
; CHECK: call void @__nasan_check_store(i64 %{{.*}}, i64 4, i64
; CHECK: store i32 1, ptr %a
entry:
  %local = alloca i32
  store i32 42, ptr %local
  store i32 1, ptr %a
  ret void
}

; Test memcpy intrinsic instrumentation
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define void @test_memcpy(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: @test_memcpy
; CHECK: call void @__nasan_check_store(i64 {{.*}}, i64 16, i64
; CHECK: call void @__nasan_check_load(i64 {{.*}}, i64 16, i64
; CHECK: call void @llvm.memcpy
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 16, i1 false)
  ret void
}

; Test memset intrinsic instrumentation
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define void @test_memset(ptr noalias %dst) {
; CHECK-LABEL: @test_memset
; CHECK: call void @__nasan_check_store(i64 {{.*}}, i64 32, i64
; CHECK: call void @llvm.memset
entry:
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 32, i1 false)
  ret void
}
