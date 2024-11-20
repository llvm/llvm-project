; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpName %[[Foo:.*]] "foo"
; CHECK-SPIRV-DAG: %[[TyChar:.*]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[TyGenPtrChar:.*]] = OpTypePointer Generic %[[TyChar]]
; CHECK-SPIRV-DAG: %[[TyFunBar:.*]] = OpTypeFunction %[[TyVoid]] %[[TyGenPtrChar]]
; CHECK-SPIRV-DAG: %[[TyLong:.*]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[TyGenPtrPtrChar:.*]] = OpTypePointer Generic %[[TyGenPtrChar]]
; CHECK-SPIRV-DAG: %[[TyFunFoo:.*]] = OpTypeFunction %[[TyVoid]] %[[TyLong]] %[[TyGenPtrPtrChar]] %[[TyGenPtrPtrChar]]
; CHECK-SPIRV-DAG: %[[TyStruct:.*]] = OpTypeStruct %[[TyLong]]
; CHECK-SPIRV-DAG: %[[Const100:.*]] = OpConstant %[[TyLong]] 100
; CHECK-SPIRV-DAG: %[[TyFunPtrGenPtrChar:.*]] = OpTypePointer Function %[[TyGenPtrChar]]
; CHECK-SPIRV-DAG: %[[TyPtrStruct:.*]] = OpTypePointer Generic %[[TyStruct]]
; CHECK-SPIRV-DAG: %[[TyPtrLong:.*]] = OpTypePointer Generic %[[TyLong]]

; CHECK-SPIRV: %[[Bar:.*]] = OpFunction %[[TyVoid]] None %[[TyFunBar]]
; CHECK-SPIRV: %[[BarArg:.*]] = OpFunctionParameter %[[TyGenPtrChar]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: OpVariable %[[TyFunPtrGenPtrChar]] Function
; CHECK-SPIRV-NEXT: OpVariable %[[TyFunPtrGenPtrChar]] Function
; CHECK-SPIRV-NEXT: OpVariable %[[TyFunPtrGenPtrChar]] Function
; CHECK-SPIRV: %[[Var1:.*]] = OpPtrCastToGeneric %[[TyGenPtrPtrChar]] %[[#]]
; CHECK-SPIRV: %[[Var2:.*]] = OpPtrCastToGeneric %[[TyGenPtrPtrChar]] %[[#]]
; CHECK-SPIRV: OpStore %[[#]] %[[BarArg]]
; CHECK-SPIRV-NEXT: OpFunctionCall %[[TyVoid]] %[[Foo]] %[[Const100]] %[[Var1]] %[[Var2]]
; CHECK-SPIRV-NEXT: OpFunctionCall %[[TyVoid]] %[[Foo]] %[[Const100]] %[[Var2]] %[[Var1]]

; CHECK-SPIRV: %[[Foo]] = OpFunction %[[TyVoid]] None %[[TyFunFoo]]
; CHECK-SPIRV-NEXT: OpFunctionParameter %[[TyLong]]
; CHECK-SPIRV-NEXT: OpFunctionParameter %[[TyGenPtrPtrChar]]
; CHECK-SPIRV-NEXT: OpFunctionParameter %[[TyGenPtrPtrChar]]

%class.CustomType = type { i64 }

define linkonce_odr dso_local spir_func void @bar(ptr addrspace(4) noundef %first) {
entry:
  %first.addr = alloca ptr addrspace(4)
  %first.addr.ascast = addrspacecast ptr %first.addr to ptr addrspace(4)
  %temp = alloca ptr addrspace(4), align 8
  %temp.ascast = addrspacecast ptr %temp to ptr addrspace(4)
  store ptr addrspace(4) %first, ptr %first.addr
  call spir_func void @foo(i64 noundef 100, ptr addrspace(4) noundef dereferenceable(8) %first.addr.ascast, ptr addrspace(4) noundef dereferenceable(8) %temp.ascast)
  call spir_func void @foo(i64 noundef 100, ptr addrspace(4) noundef dereferenceable(8) %temp.ascast, ptr addrspace(4) noundef dereferenceable(8) %first.addr.ascast)
  %var = alloca ptr addrspace(4), align 8
  ret void
}

define linkonce_odr dso_local spir_func void @foo(i64 noundef %offset, ptr addrspace(4) noundef dereferenceable(8) %in_acc1, ptr addrspace(4) noundef dereferenceable(8) %out_acc1) {
entry:
  %r0 = load ptr addrspace(4), ptr addrspace(4) %in_acc1
  %arrayidx = getelementptr inbounds %class.CustomType, ptr addrspace(4) %r0, i64 42
  %r1 = load i64, ptr addrspace(4) %arrayidx
  %r3 = load ptr addrspace(4), ptr addrspace(4) %out_acc1
  %r4 = getelementptr %class.CustomType, ptr addrspace(4) %r3, i64 43
  store i64 %r1, ptr addrspace(4) %r4
  ret void
}

