; RUN: split-file %s %t

; RUN: opt -mtriple=amdgcn-- -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %t/invalid-first-arg-addrspace.ll | FileCheck %t/invalid-first-arg-addrspace.ll
; RUN: opt -mtriple=amdgcn-- -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %t/invalid-first-arg-type.ll | FileCheck %t/invalid-first-arg-type.ll
; RUN: opt -mtriple=amdgcn-- -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %t/invalid-return.ll | FileCheck %t/invalid-return.ll
; RUN: opt -mtriple=amdgcn-- -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %t/non-variadic.ll | FileCheck %t/non-variadic.ll
; RUN: opt -mtriple=amdgcn-- -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %t/too-many-args.ll | FileCheck %t/too-many-args.ll

;--- invalid-first-arg-addrspace.ll
define amdgpu_kernel void @test_kernel(i32 %n) {
; CHECK-LABEL: define amdgpu_kernel void @test_kernel(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[MEM:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) [[STR]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    store i32 [[CALL1]], ptr addrspace(5) [[MEM]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %mem = alloca i32, align 4, addrspace(5)
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) %str, ptr addrspace(5) %str, i32 %n)
  store i32 %call1, ptr addrspace(5) %mem, align 4
  ret void
}

define i32 @test_func(i32 %n) {
; CHECK-LABEL: define i32 @test_func(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) [[STR]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret i32 [[CALL1]]
;
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) %str, ptr addrspace(5) %str, i32 %n)
  ret i32 %call1
}

define i32 @test_null_argument(i32 %n) {
; CHECK-LABEL: define i32 @test_null_argument(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) null, ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret i32 [[CALL1]]
;
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(5), ...) @printf(ptr addrspace(5) zeroinitializer, ptr addrspace(5) %str, i32 %n)
  ret i32 %call1
}

declare i32 @printf(ptr addrspace(5), ...)

;--- invalid-first-arg-type.ll
define amdgpu_kernel void @test_kernel(i32 %n) {
; CHECK-LABEL: define amdgpu_kernel void @test_kernel(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[MEM:%.*]] = alloca float, align 4, addrspace(5)
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (i32, ...) @printf(i32 [[N]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    store float [[CALL1]], ptr addrspace(5) [[MEM]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %mem = alloca float, align 4, addrspace(5)
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (i32, ...) @printf(i32 %n, ptr addrspace(5) %str, i32 %n)
    store float %call1, ptr addrspace(5) %mem, align 4
  ret void
}

define float @test_func(i32 %n) {
; CHECK-LABEL: define float @test_func(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (i32, ...) @printf(i32 [[N]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret float [[CALL1]]
;
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (i32, ...) @printf(i32 %n, ptr addrspace(5) %str, i32 %n)
  ret float %call1
}

declare float @printf(i32, ...)

;--- invalid-return.ll
@.str = private unnamed_addr addrspace(4) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
; CHECK-LABEL: define amdgpu_kernel void @test_kernel(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[MEM:%.*]] = alloca float, align 4, addrspace(5)
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) @.str, ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    store float [[CALL1]], ptr addrspace(5) [[MEM]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %mem = alloca float, align 4, addrspace(5)
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) @.str, ptr addrspace(5) %str, i32 %n)
  store float %call1, ptr addrspace(5) %mem, align 4
  ret void
}

define float @test_func(i32 %n) {
; CHECK-LABEL: define float @test_func(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) @.str, ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret float [[CALL1]]
;
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) @.str, ptr addrspace(5) %str, i32 %n)
  ret float %call1
}

define float @test_null_argument(i32 %n) {
; CHECK-LABEL: define float @test_null_argument(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) null, ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret float [[CALL1]]
;
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), ...) @printf(ptr addrspace(4) null, ptr addrspace(5) %str, i32 %n)
  ret float %call1
}

declare float @printf(ptr addrspace(4), ...)

;--- non-variadic.ll
@.str = private unnamed_addr addrspace(4) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
; CHECK-LABEL: define amdgpu_kernel void @test_kernel(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[MEM:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 @printf(ptr addrspace(4) @.str, i32 [[N]])
; CHECK-NEXT:    store i32 [[CALL1]], ptr addrspace(5) [[MEM]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %mem = alloca i32, align 4, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), i32) @printf(ptr addrspace(4) @.str, i32 %n)
  store i32 %call1, ptr addrspace(5) %mem, align 4
  ret void
}

define i32 @test_func(i32 %n) {
; CHECK-LABEL: define i32 @test_func(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 @printf(ptr addrspace(4) @.str, i32 [[N]])
; CHECK-NEXT:    ret i32 [[CALL1]]
;
entry:
  %call1 = call i32 (ptr addrspace(4), i32) @printf(ptr addrspace(4) @.str, i32 %n)
  ret i32 %call1
}

define i32 @test_null_argument(i32 %n) {
; CHECK-LABEL: define i32 @test_null_argument(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:    [[CALL1:%.*]] = call i32 @printf(ptr addrspace(4) null, i32 [[N]])
; CHECK-NEXT:    ret i32 [[CALL1]]
;
  %call1 = call i32 (ptr addrspace(4), i32) @printf(ptr addrspace(4) null, i32 %n)
  ret i32 %call1
}

declare i32 @printf(ptr addrspace(4), i32)

;--- too-many-args.ll
@.str = private unnamed_addr addrspace(4) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
; CHECK-LABEL: define amdgpu_kernel void @test_kernel(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[MEM:%.*]] = alloca float, align 4, addrspace(5)
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) @.str, i32 [[N]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    store float [[CALL1]], ptr addrspace(5) [[MEM]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %mem = alloca float, align 4, addrspace(5)
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) @.str, i32 %n, ptr addrspace(5) %str, i32 %n)
  store float %call1, ptr addrspace(5) %mem, align 4
  ret void
}

define float @test_func(i32 %n) {
; CHECK-LABEL: define float @test_func(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) @.str, i32 [[N]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret float [[CALL1]]
;
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) @.str, i32 %n, ptr addrspace(5) %str, i32 %n)
  ret float %call1
}

define float @test_null_argument(i32 %n) {
; CHECK-LABEL: define float @test_null_argument(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:    [[STR:%.*]] = alloca [9 x i8], align 1, addrspace(5)
; CHECK-NEXT:    [[CALL1:%.*]] = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) null, i32 [[N]], ptr addrspace(5) [[STR]], i32 [[N]])
; CHECK-NEXT:    ret float [[CALL1]]
;
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call float (ptr addrspace(4), i32, ...) @printf(ptr addrspace(4) null, i32 %n, ptr addrspace(5) %str, i32 %n)
  ret float %call1
}

declare float @printf(ptr addrspace(4), i32, ...)
