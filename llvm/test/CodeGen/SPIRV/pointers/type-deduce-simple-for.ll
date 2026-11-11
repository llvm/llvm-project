; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - --translator-compatibility-mode | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-COMPAT,CHECK-COMPAT64
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-DEFVER
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - --translator-compatibility-mode | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-COMPAT,CHECK-COMPAT32
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#IntConst1:]] = OpConstant %[[#IntTy]] 1
; CHECK-SPIRV-DAG: %[[#ArrTy:]] = OpTypeArray %[[#IntTy]] %[[#IntConst1]]
; CHECK-SPIRV-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#LongConst4:]] = OpConstant %[[#LongTy]] 4
; CHECK-SPIRV-DAG: %[[#IntConst123:]] = OpConstant %[[#IntTy]] 123
; CHECK-SPIRV-DAG: %[[#CharPtrTy:]] = OpTypePointer Function %[[#CharTy]]
; CHECK-SPIRV-DAG: %[[#ArrPtrTy:]] = OpTypePointer Function %[[#ArrTy]]
; CHECK-SPIRV-DAG: %[[#IntPtrTy:]] = OpTypePointer Function %[[#IntTy]]
; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#LblEntry:]] = OpLabel
; CHECK-SPIRV: %[[#Value:]] = OpVariable %[[#ArrPtrTy]] Function
; CHECK-SPIRV: %[[#ValueAsCharPtr:]] = OpBitcast %[[#CharPtrTy]] %[[#Value]]
; CHECK-SPIRV: %[[#Eof:]] = OpInBoundsPtrAccessChain %[[#CharPtrTy]] %[[#ValueAsCharPtr]] %[[#LongConst4]]
; CHECK-SPIRV: %[[#EofAsArray:]] = OpBitcast %[[#ArrPtrTy]] %[[#Eof]]
; CHECK-SPIRV: OpBranch %[[#LblCond:]]
; CHECK-SPIRV: %[[#LblCond]] = OpLabel
; CHECK-SPIRV: %[[#Iter:]] = OpPhi %[[#ArrPtrTy]] %[[#Value]] %[[#LblEntry]] %[[#CurrValue:]] %[[#LblBody:]]
; CHECK-COMPAT64: %[[#IterInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#Iter]]
; CHECK-COMPAT64: %[[#EofInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#EofAsArray]]
; CHECK-COMPAT32: %[[#IterInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#Iter]]
; CHECK-COMPAT32: %[[#EofInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#EofAsArray]]
; CHECK-COMPAT: %[[#Is:]] = OpIEqual %[[#BoolTy]] %[[#IterInt]] %[[#EofInt]]
; CHECK-DEFVER: %[[#Is:]] = OpPtrEqual %[[#BoolTy]] %[[#Iter]] %[[#EofAsArray]]
; CHECK-SPIRV: OpBranchConditional %[[#Is]] %[[#LblExit:]] %[[#LblBody]]
; CHECK-SPIRV: %[[#LblBody]] = OpLabel
; CHECK-SPIRV: %[[#IterAsIntPtr:]] = OpBitcast %[[#IntPtrTy]] %[[#Iter]]
; CHECK-SPIRV: OpStore %[[#IterAsIntPtr]] %[[#IntConst123]] Aligned 4
; CHECK-SPIRV: %[[#IterAsCharPtr:]] = OpBitcast %[[#CharPtrTy]] %[[#Iter]]
; CHECK-SPIRV: %[[#CurrValueAsCharPtr:]] = OpInBoundsPtrAccessChain %[[#CharPtrTy]] %[[#IterAsCharPtr]] %[[#LongConst4]]
; CHECK-SPIRV: %[[#CurrValue]] = OpBitcast %[[#ArrPtrTy]] %[[#CurrValueAsCharPtr]]
; CHECK-SPIRV: OpBranch %[[#LblCond]]
; CHECK-SPIRV: %[[#LblExit]] = OpLabel
; CHECK-SPIRV: OpFunctionEnd

define spir_kernel void @foo() {
entry:
  %v = alloca [1 x i32], align 4
  %eof = getelementptr inbounds i8, ptr %v, i64 4
  br label %cond

cond:
  %iter = phi ptr [ %v, %entry ], [ %curr, %body ]
  %is = icmp eq ptr %iter, %eof
  br i1 %is, label %exit, label %body

body:
  store i32 123, ptr %iter, align 4
  %curr = getelementptr inbounds i8, ptr %iter, i64 4
  br label %cond

exit:
  ret void
}

; A call result stored to null, then compared with a local address, keeps its
; OpFunctionCall type.
; CHECK-SPIRV-LABEL: Begin function store_call_result_then_icmp
; CHECK-SPIRV: %[[#StoreCall:]] = OpFunctionCall %[[#]]
; CHECK-SPIRV: OpStore %[[#]] %[[#StoreCall]] Aligned 8
; CHECK-SPIRV: %[[#StoreLoc:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK-COMPAT64: %[[#StoreLocInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#StoreLoc]]
; CHECK-COMPAT64: %[[#StoreCallInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#StoreCall]]
; CHECK-COMPAT32: %[[#StoreLocInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#StoreLoc]]
; CHECK-COMPAT32: %[[#StoreCallInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#StoreCall]]
; CHECK-COMPAT: OpINotEqual %[[#BoolTy]] %[[#StoreLocInt]] %[[#StoreCallInt]]
; CHECK-DEFVER: OpPtrNotEqual %[[#]] %[[#StoreLoc]] %[[#StoreCall]]

declare spir_func ptr addrspace(4) @f()
define spir_func i1 @store_call_result_then_icmp() {
entry:
  %v = alloca ptr addrspace(4), align 8
  %call = call spir_func ptr addrspace(4) @f()
  store ptr addrspace(4) %call, ptr addrspace(4) null, align 8
  %cast = addrspacecast ptr %v to ptr addrspace(4)
  %cmp = icmp ne ptr addrspace(4) %cast, %call
  ret i1 %cmp
}

; The same comparison, with no store.
; CHECK-SPIRV-LABEL: Begin function icmp_call_result_only
; CHECK-SPIRV: %[[#IcmpCall:]] = OpFunctionCall %[[#]]
; CHECK-SPIRV: %[[#IcmpLoc:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK-COMPAT64: %[[#IcmpLocInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#IcmpLoc]]
; CHECK-COMPAT64: %[[#IcmpCallInt:]] = OpConvertPtrToU %[[#LongTy]] %[[#IcmpCall]]
; CHECK-COMPAT32: %[[#IcmpLocInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#IcmpLoc]]
; CHECK-COMPAT32: %[[#IcmpCallInt:]] = OpConvertPtrToU %[[#IntTy]] %[[#IcmpCall]]
; CHECK-COMPAT: OpINotEqual %[[#BoolTy]] %[[#IcmpLocInt]] %[[#IcmpCallInt]]
; CHECK-DEFVER: OpPtrNotEqual %[[#]] %[[#IcmpLoc]] %[[#IcmpCall]]

define spir_func i1 @icmp_call_result_only() {
entry:
  %v = alloca ptr addrspace(4), align 8
  %call = call spir_func ptr addrspace(4) @f()
  %cast = addrspacecast ptr %v to ptr addrspace(4)
  %cmp = icmp ne ptr addrspace(4) %cast, %call
  ret i1 %cmp
}
