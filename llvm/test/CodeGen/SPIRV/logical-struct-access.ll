; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK: [[uint:%[0-9]+]] = OpTypeInt 32 0

%A = type {
  i32,
  i32
}
; CHECK:    [[A:%[0-9]+]] = OpTypeStruct [[uint]] [[uint]]

%B = type {
  %A,
  i32,
  %A
}
; CHECK:    [[B:%[0-9]+]] = OpTypeStruct [[A]] [[uint]] [[A]]

; CHECK: [[uint_0:%[0-9]+]] = OpConstant [[uint]] 0
; CHECK: [[uint_1:%[0-9]+]] = OpConstant [[uint]] 1
; CHECK: [[uint_2:%[0-9]+]] = OpConstant [[uint]] 2

; CHECK: [[ptr_uint:%[0-9]+]] = OpTypePointer Function [[uint]]
; CHECK:    [[ptr_A:%[0-9]+]] = OpTypePointer Function [[A]]
; CHECK:    [[ptr_B:%[0-9]+]] = OpTypePointer Function [[B]]

define void @main() #1 {
entry:
  %0 = alloca %B, align 4
; CHECK: [[tmp:%[0-9]+]] = OpVariable [[ptr_B]] Function

  %1 = getelementptr %B, ptr %0, i32 0, i32 0
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_A]] [[tmp]] [[uint_0]]
  %2 = getelementptr inbounds %B, ptr %0, i32 0, i32 0
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_A]] [[tmp]] [[uint_0]]

  %3 = getelementptr %B, ptr %0, i32 0, i32 1
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[tmp]] [[uint_1]]
  %4 = getelementptr inbounds %B, ptr %0, i32 0, i32 1
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_uint]] [[tmp]] [[uint_1]]

  %5 = getelementptr %B, ptr %0, i32 0, i32 2
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_A]] [[tmp]] [[uint_2]]
  %6 = getelementptr inbounds %B, ptr %0, i32 0, i32 2
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_A]] [[tmp]] [[uint_2]]

  %7 = getelementptr %B, ptr %0, i32 0, i32 2, i32 1
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[tmp]] [[uint_2]] [[uint_1]]
  %8 = getelementptr inbounds %B, ptr %0, i32 0, i32 2, i32 1
; CHECK: {{%[0-9]+}} = OpInBoundsAccessChain [[ptr_uint]] [[tmp]] [[uint_2]] [[uint_1]]

  %9 = getelementptr %B, ptr %0, i32 0, i32 2
  %10 = getelementptr %A, ptr %9, i32 0, i32 1
; CHECK: [[x:%[0-9]+]] = OpAccessChain [[ptr_A]] [[tmp]] [[uint_2]]
; CHECK:   {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[x]] [[uint_1]]

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
