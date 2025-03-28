; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:    %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#uint_pp:]] = OpTypePointer Private %[[#uint]]
; CHECK-DAG: %[[#uint_fp:]] = OpTypePointer Function %[[#uint]]
; CHECK-DAG:  %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:      %[[#v2:]] = OpTypeVector %[[#uint]] 2
; CHECK-DAG:      %[[#v3:]] = OpTypeVector %[[#uint]] 3
; CHECK-DAG:      %[[#v4:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG:   %[[#v4_pp:]] = OpTypePointer Private %[[#v4]]
; CHECK-DAG:   %[[#v4_fp:]] = OpTypePointer Function %[[#v4]]

define internal spir_func <3 x i32> @foo(ptr addrspace(10) %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 0 0

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooDefault(ptr %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 0 0

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooBounds(ptr %a) {

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#v4_fp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 0 0

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <2 x i32> @bar(ptr addrspace(10) %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  ; partial loading of a vector: v4 -> v2.
  %2 = load <2 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v2]] %[[#load]] %[[#load]] 0 0

  ret <2 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @baz(ptr addrspace(10) %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  ; Loading of the first scalar of a vector: v4 -> int.
  %2 = load i32, ptr addrspace(10) %1, align 16
; CHECK: %[[#ptr:]] = OpAccessChain %[[#uint_pp]] %[[#tmp]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 16

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @bazDefault(ptr %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#]]

  ; Loading of the first scalar of a vector: v4 -> int.
  %2 = load i32, ptr %1, align 16
; CHECK: %[[#ptr:]] = OpAccessChain %[[#uint_fp]] %[[#tmp]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 16

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @bazBounds(ptr %a) {

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#v4_fp]] %[[#]]

  ; Loading of the first scalar of a vector: v4 -> int.
  %2 = load i32, ptr %1, align 16
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#uint_fp]] %[[#tmp]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 16

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}
