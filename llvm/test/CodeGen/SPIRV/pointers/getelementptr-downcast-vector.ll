; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:    %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#uint_pp:]] = OpTypePointer Private %[[#uint]]
; CHECK-DAG: %[[#uint_fp:]] = OpTypePointer Function %[[#uint]]
; CHECK-DAG:  %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:  %[[#uint_1:]] = OpConstant %[[#uint]] 1
; CHECK-DAG:  %[[#uint_2:]] = OpConstant %[[#uint]] 2
; CHECK-DAG:      %[[#v2:]] = OpTypeVector %[[#uint]] 2
; CHECK-DAG:      %[[#v3:]] = OpTypeVector %[[#uint]] 3
; CHECK-DAG:      %[[#v4:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG:   %[[#v2_01:]] = OpConstantComposite %[[#v2]] %[[#uint_0]] %[[#uint_1]]
; CHECK-DAG:  %[[#v3_012:]] = OpConstantComposite %[[#v3]] %[[#uint_0]] %[[#uint_1]] %[[#uint_2]]
; CHECK-DAG:   %[[#v4_pp:]] = OpTypePointer Private %[[#v4]]
; CHECK-DAG:   %[[#v4_fp:]] = OpTypePointer Function %[[#v4]]

define internal spir_func <3 x i32> @foo(ptr addrspace(10) %a) {
; CHECK: %[[#foo_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#foo_a]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]]
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooDefault(ptr %a) {
; CHECK: %[[#fooD_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#fooD_a]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]]
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooBounds(ptr %a) {
; CHECK: %[[#fooB_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#v4_fp]] %[[#fooB_a]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]]
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <2 x i32> @bar(ptr addrspace(10) %a) {
; CHECK: %[[#bar_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#bar_a]]

  ; partial loading of a vector: v4 -> v2.
  %2 = load <2 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]]
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v2]] %[[#load]] %[[#load]] 0 1

  ret <2 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @baz(ptr addrspace(10) %a) {
; CHECK: %[[#baz_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
  %2 = load i32, ptr addrspace(10) %1, align 16
; CHECK: %[[#ptr:]] = OpAccessChain %[[#uint_pp]] %[[#baz_a]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]]

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @bazDefault(ptr %a) {
; CHECK: %[[#bazD_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
  %2 = load i32, ptr %1, align 16
; CHECK: %[[#ptr:]] = OpAccessChain %[[#uint_fp]] %[[#bazD_a]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]]

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @bazBounds(ptr %a) {
; CHECK: %[[#bazB_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
  %2 = load i32, ptr %1, align 16
; CHECK: %[[#ptr:]] = OpAccessChain %[[#uint_fp]] %[[#bazB_a]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]]

  ret i32 %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func void @foos(ptr addrspace(10) %a) {
; CHECK: %[[#foos_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#ptr:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#foos_a]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr addrspace(10) %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]]
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]]

  ret void
}

define internal spir_func void @foosDefault(ptr %a) {
; CHECK: %[[#foosD_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#foosD_a]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]]
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]]

  ret void
}

define internal spir_func void @foosBounds(ptr %a) {
; CHECK: %[[#foosB_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_fp]] %[[#foosB_a]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr %1, align 64
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]]
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]]

  ret void
}

define internal spir_func void @bars(ptr addrspace(10) %a) {
; CHECK: %[[#bars_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_pp]] %[[#bars_a]]

  store <2 x i32> <i32 0, i32 1>, ptr addrspace(10) %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]]
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v2_01]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v2_01]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK: OpStore %[[#ptr]] %[[#out2]]

  ret void
}

define internal spir_func void @bazs(ptr addrspace(10) %a) {
; CHECK: %[[#bazs_a:]] = OpFunctionParameter %[[#v4_pp]]

  %1 = getelementptr <4 x i32>, ptr addrspace(10) %a, i64 0
  store i32 0, ptr addrspace(10) %1, align 32
; CHECK: %[[#tmp:]] = OpAccessChain %[[#uint_pp]] %[[#bazs_a]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]]

  ret void
}

define internal spir_func void @bazsDefault(ptr %a) {
; CHECK: %[[#bazsD_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
  store i32 0, ptr %1, align 16
; CHECK: %[[#tmp:]] = OpInBoundsAccessChain %[[#uint_fp]] %[[#bazsD_a]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]]

  ret void
}

define internal spir_func void @bazsBounds(ptr %a) {
; CHECK: %[[#bazsB_a:]] = OpFunctionParameter %[[#v4_fp]]

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
  store i32 0, ptr %1, align 16
; CHECK: %[[#tmp:]] = OpAccessChain %[[#uint_fp]] %[[#bazsB_a]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]]

  ret void
}
