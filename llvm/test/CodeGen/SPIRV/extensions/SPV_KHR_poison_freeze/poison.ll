; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --check-prefixes=CHECK,OCL
; RUN: llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --check-prefixes=CHECK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability PoisonFreezeKHR
; CHECK-DAG: OpExtension "SPV_KHR_poison_freeze"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V4F:]] = OpTypeVector %[[#F32]] 4

; CHECK-DAG: %[[#PI:]] = OpPoisonKHR %[[#I32]]
; CHECK-DAG: %[[#PV:]] = OpPoisonKHR %[[#V4]]
; CHECK-DAG: %[[#PVF:]] = OpPoisonKHR %[[#V4F]]
; CHECK-DAG: %[[#PF:]] = OpPoisonKHR %[[#F32]]
; CHECK-DAG: %[[#PA:]] = OpPoisonKHR %[[#ARR]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#PI]]
; CHECK: OpStore %[[#]] %[[#PI]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#PV]]
; CHECK: OpStore %[[#]] %[[#PV]]

; CHECK: OpFunction
; CHECK: OpStore %[[#]] %[[#PA]]

; CHECK: OpFunction
; CHECK: OpVectorShuffle %[[#V4F]] %[[#]] %[[#PVF]]

; CHECK: OpFunction
; CHECK: OpSelect %[[#V4F]] %[[#]] %[[#]] %[[#PVF]]

; CHECK: OpFunction
; CHECK: OpFAdd %[[#F32]] %[[#]] %[[#PF]]

; CHECK: OpFunction
; CHECK: OpFAdd %[[#F32]] %[[#]] %[[#PF]]

; OpPhi survives only on the OpenCL flow. The Vulkan path structurizes the
; CFG and folds phi(%v, poison) to %v before lowering.
; OCL: OpFunction
; OCL: OpPhi %[[#V4F]] %[[#]] %[[#]] %[[#PVF]] %[[#]]

%arr = type [4 x i32]

define void @poison_scalar(ptr %dst) {
  store i32 poison, ptr %dst
  store i32 poison, ptr %dst
  ret void
}

define void @poison_vector(ptr %dst) {
  store <4 x i32> poison, ptr %dst
  store <4 x i32> poison, ptr %dst
  ret void
}

define void @poison_aggregate(ptr %dst) {
  store %arr poison, ptr %dst
  ret void
}

define <4 x float> @poison_shuffle(float %val) {
  %splat = insertelement <4 x float> poison, float %val, i32 0
  %bcast = shufflevector <4 x float> %splat, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %bcast
}

define <4 x float> @poison_select(i1 %cond, <4 x float> %v) {
  %r = select i1 %cond, <4 x float> %v, <4 x float> poison
  ret <4 x float> %r
}

define float @poison_fadd(float %x) {
  %r = fadd float %x, poison
  ret float %r
}

; A direct call to the type-overloaded intrinsic must lower the same way.
declare float @llvm.spv.poison.f32()

define float @poison_intrinsic_call(float %x) {
  %p = call float @llvm.spv.poison.f32()
  %r = fadd float %x, %p
  ret float %r
}

define <4 x float> @poison_phi(i1 %c, <4 x float> %v) {
entry:
  br i1 %c, label %a, label %b
a:
  br label %j
b:
  br label %j
j:
  %p = phi <4 x float> [ %v, %a ], [ poison, %b ]
  ret <4 x float> %p
}
