; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s

; CHECK-DAG: [[uint:%[0-9]+]] = OpTypeInt 32 0

%A = type {
  i32,
  i32
}
; CHECK-DAG:    [[A:%[0-9]+]] = OpTypeStruct [[uint]] [[uint]]

%B = type {
  %A,
  i32,
  %A
}
; CHECK-DAG:    [[B:%[0-9]+]] = OpTypeStruct [[A]] [[uint]] [[A]]

; CHECK-DAG: [[uint_0:%[0-9]+]] = OpConstant [[uint]] 0
; CHECK-DAG: [[uint_1:%[0-9]+]] = OpConstant [[uint]] 1
; CHECK-DAG: [[uint_2:%[0-9]+]] = OpConstant [[uint]] 2

; CHECK-DAG: [[ptr_uint:%[0-9]+]] = OpTypePointer Function [[uint]]
; CHECK-DAG:    [[ptr_A:%[0-9]+]] = OpTypePointer Function [[A]]
; CHECK-DAG:    [[ptr_B:%[0-9]+]] = OpTypePointer Function [[B]]

define internal ptr @gep_B_0(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_A]] [[tmp]] [[uint_0]]
  %res = getelementptr %B, ptr %base, i32 0, i32 0
  ret ptr %res
}

define internal ptr @gep_inbounds_B_0(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_A]] [[tmp]] [[uint_0]]
  %res = getelementptr inbounds %B, ptr %base, i32 0, i32 0
  ret ptr %res
}

define internal ptr @gep_B_1(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[tmp]] [[uint_1]]
  %res = getelementptr %B, ptr %base, i32 0, i32 1
  ret ptr %res
}

define internal ptr @gep_inbounds_B_1(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_uint]] [[tmp]] [[uint_1]]
  %res = getelementptr inbounds %B, ptr %base, i32 0, i32 1
  ret ptr %res
}

define internal ptr @gep_B_2(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_A]] [[tmp]] [[uint_2]]
  %res = getelementptr %B, ptr %base, i32 0, i32 2
  ret ptr %res
}

define internal ptr @gep_inbounds_B_2(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_A]] [[tmp]] [[uint_2]]
  %res = getelementptr inbounds %B, ptr %base, i32 0, i32 2
  ret ptr %res
}

define internal ptr @gep_B_2_1(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[tmp]] [[uint_2]] [[uint_1]]
  %res = getelementptr %B, ptr %base, i32 0, i32 2, i32 1
  ret ptr %res
}

define internal ptr @gep_inbounds_B_2_1(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_uint]] [[tmp]] [[uint_2]] [[uint_1]]
  %res = getelementptr inbounds %B, ptr %base, i32 0, i32 2, i32 1
  ret ptr %res
}

define internal ptr @gep_B_2_A_1(ptr %base) {
; CHECK: [[tmp:%[0-9]+]] = OpFunctionParameter [[ptr_B]]
; CHECK: [[x:%[0-9]+]] = OpAccessChain [[ptr_A]] [[tmp]] [[uint_2]]
; CHECK:   {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[x]] [[uint_1]]
  %x = getelementptr %B, ptr %base, i32 0, i32 2
  %res = getelementptr %A, ptr %x, i32 0, i32 1
  ret ptr %res
}

define void @main() #1 {
entry:
  %0 = alloca %B, align 4
; CHECK: [[tmp:%[0-9]+]] = OpVariable [[ptr_B]] Function

  %1 = call ptr @gep_B_0(ptr %0)
  %2 = call ptr @gep_inbounds_B_0(ptr %0)
  %3 = call ptr @gep_B_1(ptr %0)
  %4 = call ptr @gep_inbounds_B_1(ptr %0)
  %5 = call ptr @gep_B_2(ptr %0)
  %6 = call ptr @gep_inbounds_B_2(ptr %0)
  %7 = call ptr @gep_B_2_1(ptr %0)
  %8 = call ptr @gep_inbounds_B_2_1(ptr %0)
  %10 = call ptr @gep_B_2_A_1(ptr %0)

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
