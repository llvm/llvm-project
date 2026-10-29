; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

; CHECK: OpName %[[#RECUR_INNER_1:]] "recur.inner.1"
; CHECK: OpName %[[#RECUR_INNER_2:]] "recur.inner.2"
; CHECK: OpName %[[#RECUR_OUTER:]] "recur.outer"
; CHECK: OpName %[[#CALLER:]] "caller"
; CHECK: OpName %[[#FOO:]] "foo"
; CHECK: %[[#INT32_TY:]] = OpTypeInt 32
; CHECK: %[[#I32PTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#INT32_TY]]
; CHECK: %[[#INT8_TY:]] = OpTypeInt 8
; CHECK: %[[#I8PTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#INT8_TY]]

;	CHECK: %[[#RECUR_INNER_1]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
;	CHECK: %[[#X:]] = OpFunctionParameter %[[#I32PTR_ADDRSPACE_7]]
;	CHECK: %[[#RET0:]] = OpBitcast %[[#I8PTR_ADDRSPACE_7]] %[[#X]]
;	CHECK: %[[#RET1:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#RECUR_INNER_2]] %[[#]] %[[#X]]
;	CHECK: %[[#RET:]] = OpPhi %[[#I8PTR_ADDRSPACE_7]] %[[#RET1]] %[[#]] %[[#RET0]] %[[#]]
;	CHECK: OpReturnValue %[[#RET]]
define spir_func ptr addrspace(7) @recur.inner.1(ptr addrspace(7) %x, i32 %v) {
bb:
  %isBase = icmp sgt i32 %v, 0
  br i1 %isBase, label %recur, label %else
recur:
  %dec = sub i32 %v, 1
  %inc = call ptr addrspace(7) @recur.inner.2(i32 %dec, ptr addrspace(7) %x)
  br label %end
else:
  br label %end
end:
  %ret = phi ptr addrspace(7) [%inc, %recur], [%x, %else]
  ret ptr addrspace(7) %ret
}

;	CHECK: %[[#RECUR_INNER_2]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
;	CHECK: %[[#X1:]] = OpFunctionParameter %[[#I32PTR_ADDRSPACE_7]]
;	CHECK: %[[#GEP:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#X1]]
;	CHECK: %[[#RET2:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#RECUR_INNER_1]] %[[#GEP]]
;	CHECK: OpReturnValue %[[#RET2]]
define spir_func ptr addrspace(7) @recur.inner.2(i32 %v, ptr addrspace(7) %x) {
  %inc = getelementptr i32, ptr addrspace(7) %x, i32 1
  %ret = call ptr addrspace(7) @recur.inner.1(ptr addrspace(7) %inc, i32 %v)
  ret ptr addrspace(7) %ret
}

;	CHECK: %[[#RECUR_OUTER]] = OpFunction
;	CHECK: %[[#X2:]] = OpFunctionParameter %[[#I32PTR_ADDRSPACE_7]]
; CHECK: %[[#ARG:]] = OpFunctionParameter %[[#I32PTR_GENERIC:]]
; CHECK: %[[#LD:]] = OpLoad %[[#INT32_TY]] %[[#ARG]]
; CHECK: %[[#STV:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#RECUR_INNER_1]] %[[#X2]] %[[#LD]]
; CHECK: %[[#ARG_AS_I8PTR_ADDRSPACE_7PTR_GENERIC:]] = OpBitcast %[[#I8PTR_ADDRSPACE_7PTR_GENERIC:]] %[[#ARG]]
; CHECK: OpStore %[[#ARG_AS_I8PTR_ADDRSPACE_7PTR_GENERIC]] %[[#STV]]
define spir_func void @recur.outer(ptr addrspace(7) %x, ptr addrspace(4) %arg) {
  %bound = load i32, ptr addrspace(4) %arg
  %ret = call ptr addrspace(7) @recur.inner.1(ptr addrspace(7) %x, i32 %bound)
  store ptr addrspace(7) %ret, ptr addrspace(4) %arg
  ret void
}

;	CHECK: %[[#CALLER]] = OpFunction
;	CHECK: %[[#ARG1:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_7PTR_ADDRSPACE_7:]]
; CHECK: %[[#ARG1_AS_I8PTR_ADDRSPACE_7:]] = OpBitcast %[[#I8PTR_ADDRSPACE_7]] %[[#ARG1]]
; CHECK: %[[#STV:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#EXTERN:]] %[[#ARG1_AS_I8PTR_ADDRSPACE_7]]
; CHECK: OpStore %[[#ARG1]] %[[#STV]]
declare spir_func ptr addrspace(7) @extern(ptr addrspace(7) %arg)
define spir_func void @caller(ptr addrspace(7) noundef nonnull %arg) {
  %v = call ptr addrspace(7) @extern(ptr addrspace(7) %arg)
  store ptr addrspace(7) %v, ptr addrspace(7) %arg
  ret void
}

;	CHECK: %[[#FOO]] = OpFunction %[[#I32PTR_ADDRSPACE_7]]
;	CHECK: %[[#ARG2:]] = OpFunctionParameter %[[#I32PTR_ADDRSPACE_7]]
; CHECK: %[[#RET3:]] = OpInBoundsPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#ARG2]]
; CHECK: OpReturnValue %[[#RET3]]
define spir_func  ptr addrspace(7) @foo(ptr addrspace(7) noalias noundef nonnull %arg) {
  %ret = getelementptr inbounds i32, ptr addrspace(7) %arg, i32 1
  ret ptr addrspace(7) %ret
}
