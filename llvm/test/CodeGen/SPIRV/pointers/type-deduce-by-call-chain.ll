; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpName %[[ArgCum:.*]] "_arg_cum"
; CHECK-SPIRV-DAG: OpName %[[FunTest:.*]] "test"
; CHECK-SPIRV-DAG: OpName %[[Addr:.*]] "addr"
; CHECK-SPIRV-DAG: OpName %[[StubObj:.*]] "stub_object"
; CHECK-SPIRV-DAG: OpName %[[MemOrder:.*]] "mem_order"
; CHECK-SPIRV-DAG: OpName %[[FooStub:.*]] "foo_stub"
; CHECK-SPIRV-DAG: OpName %[[FooObj:.*]] "foo_object"
; CHECK-SPIRV-DAG: OpName %[[FooMemOrder:.*]] "mem_order"
; CHECK-SPIRV-DAG: OpName %[[FooFunc:.*]] "foo"
; CHECK-SPIRV-DAG: %[[TyLong:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[TyPtrLong:.*]] = OpTypePointer CrossWorkgroup %[[TyLong]]
; CHECK-SPIRV-DAG: %[[TyFunPtrLong:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtrLong]]
; CHECK-SPIRV-DAG: %[[TyGenPtrLong:.*]] = OpTypePointer Generic %[[TyLong]]
; CHECK-SPIRV-DAG: %[[TyFunGenPtrLongLong:.*]] = OpTypeFunction %[[TyVoid]] %[[TyGenPtrLong]] %[[TyLong]]
; CHECK-SPIRV-DAG: %[[Const3:.*]] = OpConstant %[[TyLong]] 3
; CHECK-SPIRV: %[[FunTest]] = OpFunction %[[TyVoid]] None %[[TyFunPtrLong]]
; CHECK-SPIRV: %[[ArgCum]] = OpFunctionParameter %[[TyPtrLong]]
; CHECK-SPIRV: OpFunctionCall %[[TyVoid]] %[[FooFunc]] %[[Addr]] %[[Const3]]
; CHECK-SPIRV: %[[FooStub]] = OpFunction %[[TyVoid]] None %[[TyFunGenPtrLongLong]]
; CHECK-SPIRV: %[[StubObj]] = OpFunctionParameter %[[TyGenPtrLong]]
; CHECK-SPIRV: %[[MemOrder]] = OpFunctionParameter %[[TyLong]]
; CHECK-SPIRV: %[[FooFunc]] = OpFunction %[[TyVoid]] None %[[TyFunGenPtrLongLong]]
; CHECK-SPIRV: %[[FooObj]] = OpFunctionParameter %[[TyGenPtrLong]]
; CHECK-SPIRV: %[[FooMemOrder]] = OpFunctionParameter %[[TyLong]]
; CHECK-SPIRV: OpFunctionCall %[[TyVoid]] %[[FooStub]] %[[FooObj]] %[[FooMemOrder]]

define spir_kernel void @test(ptr addrspace(1) noundef align 4 %_arg_cum) {
entry:
  %lptr = getelementptr inbounds i32, ptr addrspace(1) %_arg_cum, i64 1
  %addr = addrspacecast ptr addrspace(1) %lptr to ptr addrspace(4)
  %object = bitcast ptr addrspace(4) %addr to ptr addrspace(4)
  call spir_func void @foo(ptr addrspace(4) %object, i32 3)
  ret void
}

define void @foo_stub(ptr addrspace(4) noundef %stub_object, i32 noundef %mem_order) {
entry:
  %object.addr = alloca ptr addrspace(4)
  %object.addr.ascast = addrspacecast ptr %object.addr to ptr addrspace(4)
  store ptr addrspace(4) %stub_object, ptr addrspace(4) %object.addr.ascast
  ret void
}

define void @foo(ptr addrspace(4) noundef %foo_object, i32 noundef %mem_order) {
  tail call void @foo_stub(ptr addrspace(4) noundef %foo_object, i32 noundef %mem_order)
  ret void
}

