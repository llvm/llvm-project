; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Size3:]] = OpConstant %[[#Int]] 3
; CHECK-SPIRV-DAG: %[[#Arr3:]] = OpTypeArray %[[#Char]] %[[#Size3]]
; CHECK-SPIRV-DAG: %[[#Size16:]] = OpConstant %[[#Int]] 16
; CHECK-SPIRV-DAG: %[[#Arr16:]] = OpTypeArray %[[#Char]] %[[#Size16]]
; CHECK-SPIRV-DAG: %[[#Const3:]] = OpConstant %[[#Long]] 3
; CHECK-SPIRV-DAG: %[[#One:]] = OpConstant %[[#Char]] 1
; CHECK-SPIRV-DAG: %[[#One3:]] = OpConstantComposite %[[#Arr3]] %[[#One]] %[[#One]] %[[#One]]
; CHECK-SPIRV-DAG: %[[#Zero3:]] = OpConstantNull %[[#Arr3]]
; CHECK-SPIRV-DAG: %[[#Const16:]] = OpConstant %[[#Long]] 16
; CHECK-SPIRV-DAG: %[[#One16:]] = OpConstantComposite %[[#Arr16]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]]
; CHECK-SPIRV-DAG: %[[#Zero16:]] = OpConstantNull %[[#Arr16]]

; The first set of functions.
; CHECK-SPIRV-DAG: %[[#PtrArr3:]] = OpTypePointer UniformConstant %[[#Arr3]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr3]] UniformConstant %[[#One3]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr3]] UniformConstant %[[#Zero3]]
; CHECK-SPIRV-DAG: %[[#PtrArr16:]] = OpTypePointer UniformConstant %[[#Arr16]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr16]] UniformConstant %[[#One16]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr16]] UniformConstant %[[#Zero16]]

; The second set of functions.
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr3]] UniformConstant %[[#One3]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr3]] UniformConstant %[[#Zero3]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr16]] UniformConstant %[[#One16]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArr16]] UniformConstant %[[#Zero16]]

%Vec3 = type { <3 x i8> }
%Vec16 = type { <16 x i8> }

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const3]] Aligned 4
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const3]] Aligned 4
; CHECK-SPIRV: OpFunctionEnd
define spir_kernel void @foo(ptr addrspace(1) noundef align 16 %arg) {
  %a1 = getelementptr inbounds %Vec3, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a1, i8 0, i64 3, i1 false)
  %a2 = getelementptr inbounds %Vec3, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a2, i8 1, i64 3, i1 false)
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const16]] Aligned 4
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const16]] Aligned 4
; CHECK-SPIRV: OpFunctionEnd
define spir_kernel void @bar(ptr addrspace(1) noundef align 16 %arg) {
  %a1 = getelementptr inbounds %Vec16, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a1, i8 0, i64 16, i1 false)
  %a2 = getelementptr inbounds %Vec16, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a2, i8 1, i64 16, i1 false)
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const3]] Aligned 4
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const3]] Aligned 4
; CHECK-SPIRV: OpFunctionEnd
define spir_kernel void @foo_2(ptr addrspace(1) noundef align 16 %arg) {
  %a1 = getelementptr inbounds %Vec3, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a1, i8 0, i64 3, i1 false)
  %a2 = getelementptr inbounds %Vec3, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a2, i8 1, i64 3, i1 false)
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const16]] Aligned 4
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#]] %[[#Const16]] Aligned 4
; CHECK-SPIRV: OpFunctionEnd
define spir_kernel void @bar_2(ptr addrspace(1) noundef align 16 %arg) {
  %a1 = getelementptr inbounds %Vec16, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a1, i8 0, i64 16, i1 false)
  %a2 = getelementptr inbounds %Vec16, ptr addrspace(1) %arg, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %a2, i8 1, i64 16, i1 false)
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1) nocapture writeonly, i8, i64, i1 immarg)
