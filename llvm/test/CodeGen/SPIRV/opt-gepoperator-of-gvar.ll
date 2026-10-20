; RUN: llc -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O2 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrChar:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PtrInt:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#C648:]] = OpConstant %[[#]] 648
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-DAG: %[[#VarInit:]] = OpConstantNull %[[#Struct]]
; CHECK-DAG: %[[#PtrStruct:]] = OpTypePointer CrossWorkgroup %[[#Struct]]
; CHECK-DAG: %[[#Var:]] = OpVariable %[[#PtrStruct]] CrossWorkgroup %[[#VarInit]]
; CHECK-DAG: %[[#Bytes:]] = OpVariable %[[#PtrChar]] CrossWorkgroup %[[#]]
; CHECK-DAG: %[[#BytesGEP:]] = OpSpecConstantOp %[[#PtrChar]] InBoundsPtrAccessChain %[[#Bytes]] %[[#C648]]

; CHECK: OpFunction
; CHECK: %[[#]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Line:]] = OpFunctionParameter %[[#Int]]
; CHECK: %[[#]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Casted:]] = OpBitcast %[[#PtrChar]] %[[#Var]]
; CHECK: %[[#AddrChar:]] = OpInBoundsPtrAccessChain %[[#PtrChar]] %[[#Casted]] %[[#C648]]
; CHECK: %[[#AddrInt:]] = OpBitcast %[[#PtrInt]] %[[#AddrChar]]
; CHECK: OpStore %[[#AddrInt]] %[[#Line]]

%struct = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
@Mem = linkonce_odr dso_local addrspace(1) global %struct zeroinitializer, align 8

define weak dso_local spir_func void @foo(ptr addrspace(4) noundef %expr, i32 noundef %line, i1 %fl) {
entry:
  %cmp = icmp eq i32 %line, 0
  br i1 %cmp, label %lbl, label %exit

lbl:
  store i32 %line, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @Mem, i64 648), align 8
  br i1 %fl, label %lbl, label %exit

exit:
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Line2:]] = OpFunctionParameter %[[#Int]]
; CHECK: %[[#]] = OpFunctionParameter %[[#]]
; CHECK: %[[#AddrInt2:]] = OpBitcast %[[#PtrInt]] %[[#BytesGEP]]
; CHECK: OpStore %[[#AddrInt2]] %[[#Line2]]

@Bytes = linkonce_odr dso_local addrspace(1) global i8 zeroinitializer, align 8

define weak dso_local spir_func void @bar(ptr addrspace(4) noundef %expr, i32 noundef %line, i1 %fl) {
entry:
  %cmp = icmp eq i32 %line, 0
  br i1 %cmp, label %lbl, label %exit

lbl:
  store i32 %line, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @Bytes, i64 648), align 8
  br i1 %fl, label %lbl, label %exit

exit:
  ret void
}
