; Verify that aggregate PHI nodes in loops are correctly lowered to
; OpPhi with composite types, without introducing type mismatches
; between the PHI result and its incoming values.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ARRTY:]] = OpTypeArray
; CHECK-DAG: %[[#ARRVECTY:]] = OpTypeArray
; CHECK-DAG: %[[#STRUCTTY:]] = OpTypeStruct

; CHECK-DAG: %[[#PHI1:]] = OpPhi %[[#ARRTY]]
; CHECK-DAG: %[[#]] = OpPhi %[[#]]
; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#PHI1]]
; CHECK: %[[#]] = OpFAdd
; CHECK: %[[#]] = OpCompositeInsert %[[#ARRTY]]
define void @loop_phi_single_element_array_scalar(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %agg = phi [1 x float] [ %agg.new, %loop ], [ zeroinitializer, %entry ]
  %prev = extractvalue [1 x float] %agg, 0
  %sum = fadd float %prev, 1.0
  %agg.new = insertvalue [1 x float] poison, float %sum, 0
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, 64
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue [1 x float] %agg, 0
  store float %final, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-DAG: %[[#PHI2:]] = OpPhi %[[#ARRVECTY]]
; CHECK-DAG: %[[#]] = OpPhi %[[#]]
; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#PHI2]]
; CHECK: %[[#]] = OpFAdd
; CHECK: %[[#]] = OpCompositeInsert %[[#ARRVECTY]]
define void @loop_phi_single_element_array_vector(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %agg = phi [1 x <4 x float>] [ %agg.new, %loop ], [ zeroinitializer, %entry ]
  %prev = extractvalue [1 x <4 x float>] %agg, 0
  %sum = fadd <4 x float> %prev, <float 1.0, float 1.0, float 1.0, float 1.0>
  %agg.new = insertvalue [1 x <4 x float>] poison, <4 x float> %sum, 0
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, 64
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue [1 x <4 x float>] %agg, 0
  store <4 x float> %final, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK-DAG: %[[#PHI3:]] = OpPhi %[[#ARRTY]]
; CHECK-DAG: %[[#]] = OpPhi %[[#]]
; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#PHI3]]
; CHECK: %[[#]] = OpFAdd
; CHECK: %[[#]] = OpCompositeInsert %[[#ARRTY]]
define void @loop_phi_insert_into_agg(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %agg = phi [1 x float] [ %agg.new, %loop ], [ zeroinitializer, %entry ]
  %prev = extractvalue [1 x float] %agg, 0
  %sum = fadd float %prev, 1.0
  %agg.new = insertvalue [1 x float] %agg, float %sum, 0
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, 64
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue [1 x float] %agg, 0
  store float %final, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-DAG: %[[#PHI4:]] = OpPhi %[[#STRUCTTY]]
; CHECK-DAG: %[[#]] = OpPhi %[[#]]
; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#PHI4]]
; CHECK: %[[#]] = OpFAdd
; CHECK: %[[#]] = OpCompositeInsert %[[#STRUCTTY]]
define void @loop_phi_struct(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %agg = phi {float} [ %agg.new, %loop ], [ zeroinitializer, %entry ]
  %prev = extractvalue {float} %agg, 0
  %sum = fadd float %prev, 1.0
  %agg.new = insertvalue {float} poison, float %sum, 0
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, 64
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue {float} %agg, 0
  store float %final, ptr addrspace(1) %out, align 4
  ret void
}
