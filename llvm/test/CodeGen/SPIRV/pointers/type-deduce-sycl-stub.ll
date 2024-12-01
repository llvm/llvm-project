; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=all --translator-compatibility-mode --avoid-spirv-capabilities=Shader %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=all --translator-compatibility-mode --avoid-spirv-capabilities=Shader %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpName %[[#F:]] "finish"
; CHECK-SPIRV-DAG: OpName %[[#FH:]] "finish_helper"
; CHECK-SPIRV-DAG: OpName %[[#S:]] "start"
; CHECK-SPIRV-DAG: OpName %[[#SH:]] "start_helper"

; CHECK-SPIRV-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#FPtrLong:]] = OpTypePointer Function %[[#Long]]
; CHECK-SPIRV-DAG: %[[#GPtrLong:]] = OpTypePointer Generic %[[#Long]]
; CHECK-SPIRV-DAG: %[[#C3:]] = OpConstant %[[#]] 3
; CHECK-SPIRV-DAG: %[[#Array3:]] = OpTypeArray %[[#Long]] %[[#C3]]
; CHECK-SPIRV-DAG: %[[#PtrArray3:]] = OpTypePointer Generic %[[#Array3]]
; CHECK-SPIRV-DAG: %[[#FPtrPtrArray3:]] = OpTypePointer Function %[[#PtrArray3]]
; CHECK-SPIRV-DAG: %[[#GPtrPtrArray3:]] = OpTypePointer Generic %[[#PtrArray3]]

; CHECK-SPIRV: %[[#FH]] = OpFunction
; CHECK-SPIRV: %[[#Arg1:]] = OpFunctionParameter %[[#PtrArray3]]
; CHECK-SPIRV: %[[#Arg2:]] = OpFunctionParameter %[[#Long]]
; CHECK-SPIRV: %[[#GrpIdAddr:]] = OpVariable %[[#FPtrPtrArray3]] Function
; CHECK-SPIRV: %[[#WIId:]] = OpVariable %[[#FPtrLong]] Function
; CHECK-SPIRV: %[[#GenGrpIdAddr:]] = OpPtrCastToGeneric %[[#GPtrPtrArray3]] %[[#GrpIdAddr]]
; CHECK-SPIRV: %[[#GenWIId:]] = OpPtrCastToGeneric %[[#GPtrLong]] %[[#WIId]]
; CHECK-SPIRV: OpStore %[[#GenGrpIdAddr]] %[[#Arg1]]
; CHECK-SPIRV: OpStore %[[#GenWIId]] %[[#Arg2]]
; CHECK-SPIRV: OpReturn
; CHECK-SPIRV: OpFunctionEnd

@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @foo() {
entry:
  call spir_func void @start()
  call spir_func void @finish()
  ret void
}

define dso_local spir_func void @start() {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0)
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  %r0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %r1 = extractelement <3 x i64> %r0, i64 0
  store i64 %r1, ptr %GroupID, align 8
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %r2 = extractelement <3 x i64> %r0, i64 1
  store i64 %r2, ptr %arrayinit.element, align 8
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %r3 = extractelement <3 x i64> %r0, i64 2
  store i64 %r3, ptr %arrayinit.element1, align 8
  %r4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8
  %r5 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %r6 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 8), align 8
  %mul = mul i64 %r5, %r6
  %r7 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 16), align 16
  %mul2 = mul i64 %mul, %r7
  %conv = trunc i64 %mul2 to i32
  call spir_func void @start_helper(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %r4, i32 noundef %conv)
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

define dso_local spir_func void @finish() {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0)
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  %r0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %r1 = extractelement <3 x i64> %r0, i64 0
  store i64 %r1, ptr %GroupID, align 8
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %r2 = extractelement <3 x i64> %r0, i64 1
  store i64 %r2, ptr %arrayinit.element, align 8
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %r3 = extractelement <3 x i64> %r0, i64 2
  store i64 %r3, ptr %arrayinit.element1, align 8
  %r4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8
  call spir_func void @finish_helper(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %r4)
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

define dso_local spir_func void @start_helper(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id, i32 noundef %wg_size) {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %wg_size.addr = alloca i32, align 4
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  %wg_size.addr.ascast = addrspacecast ptr %wg_size.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8
  store i32 %wg_size, ptr addrspace(4) %wg_size.addr.ascast, align 4
  ret void
}

define dso_local spir_func void @finish_helper(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id) {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8
  ret void
}

declare dso_local spir_func signext i8 @__spirv_SpecConstant(i32 noundef, i8 noundef signext)
