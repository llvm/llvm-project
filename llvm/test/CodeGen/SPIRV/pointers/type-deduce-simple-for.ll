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
