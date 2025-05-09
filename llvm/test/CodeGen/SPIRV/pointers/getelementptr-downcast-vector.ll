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

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooDefault(ptr %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <3 x i32> @fooBounds(ptr %a) {

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#v4_fp]] %[[#]]

  ; partial loading of a vector: v4 -> v3.
  %2 = load <3 x i32>, ptr %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v3]] %[[#load]] %[[#load]] 0 1 2

  ret <3 x i32> %2
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func <2 x i32> @bar(ptr addrspace(10) %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  ; partial loading of a vector: v4 -> v2.
  %2 = load <2 x i32>, ptr addrspace(10) %1, align 16
; CHECK: %[[#load:]] = OpLoad %[[#v4]] %[[#tmp]] Aligned 16
; CHECK: %[[#val:]] = OpVectorShuffle %[[#v2]] %[[#load]] %[[#load]] 0 1

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

define internal spir_func void @foos(ptr addrspace(10) %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#ptr:]]  = OpInBoundsAccessChain %[[#v4_pp]] %[[#]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr addrspace(10) %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]] Aligned 16
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]] Aligned 16

  ret void
}

define internal spir_func void @foosDefault(ptr %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]] Aligned 16
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]] Aligned 16

  ret void
}

define internal spir_func void @foosBounds(ptr %a) {

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_fp]] %[[#]]

  store <3 x i32> <i32 0, i32 1, i32 2>, ptr %1, align 64
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]] Aligned 64
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK:    %[[#C:]] = OpCompositeExtract %[[#uint]] %[[#v3_012]] 2
; CHECK: %[[#out3:]] = OpCompositeInsert %[[#v4]] %[[#C]] %[[#out2]] 2
; CHECK: OpStore %[[#ptr]] %[[#out3]] Aligned 64

  ret void
}

define internal spir_func void @bars(ptr addrspace(10) %a) {

  %1 = getelementptr <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_pp]] %[[#]]

  store <2 x i32> <i32 0, i32 1>, ptr addrspace(10) %1, align 16
; CHECK: %[[#out0:]] = OpLoad %[[#v4]] %[[#ptr]] Aligned 16
; CHECK:    %[[#A:]] = OpCompositeExtract %[[#uint]] %[[#v2_01]] 0
; CHECK: %[[#out1:]] = OpCompositeInsert %[[#v4]] %[[#A]] %[[#out0]] 0
; CHECK:    %[[#B:]] = OpCompositeExtract %[[#uint]] %[[#v2_01]] 1
; CHECK: %[[#out2:]] = OpCompositeInsert %[[#v4]] %[[#B]] %[[#out1]] 1
; CHECK: OpStore %[[#ptr]] %[[#out2]] Aligned 1

  ret void
}

define internal spir_func void @bazs(ptr addrspace(10) %a) {

  %1 = getelementptr <4 x i32>, ptr addrspace(10) %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_pp]] %[[#]]

  store i32 0, ptr addrspace(10) %1, align 32
; CHECK:  %[[#tmp:]] = OpInBoundsAccessChain %[[#uint_pp]] %[[#ptr]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]] Aligned 32

  ret void
}

define internal spir_func void @bazsDefault(ptr %a) {

  %1 = getelementptr inbounds <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpInBoundsAccessChain %[[#v4_fp]] %[[#]]

  store i32 0, ptr %1, align 16
; CHECK:  %[[#tmp:]] = OpInBoundsAccessChain %[[#uint_fp]] %[[#ptr]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]] Aligned 16

  ret void
}

define internal spir_func void @bazsBounds(ptr %a) {

  %1 = getelementptr <4 x i32>, ptr %a, i64 0
; CHECK: %[[#ptr:]]  = OpAccessChain %[[#v4_fp]] %[[#]]

  store i32 0, ptr %1, align 16
; CHECK:  %[[#tmp:]] = OpInBoundsAccessChain %[[#uint_fp]] %[[#ptr]] %[[#uint_0]]
; CHECK: OpStore %[[#tmp]] %[[#uint_0]] Aligned 16

  ret void
}
