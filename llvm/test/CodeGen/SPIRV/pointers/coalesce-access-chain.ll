; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.Inner = type { i32, <4 x float> }
%struct.Outer = type { [5 x %struct.Inner], i32 }

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Inner:]] = OpTypeStruct %[[#int]] %[[#vec4]]
; CHECK-DAG: %[[#idx_5:]] = OpConstant %[[#int]] 5
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Inner]] %[[#idx_5]]
; CHECK-DAG: %[[#Outer:]] = OpTypeStruct %[[#Array]] %[[#int]]
; CHECK-DAG: %[[#ptr_Outer:]] = OpTypePointer Function %[[#Outer]]
; CHECK-DAG: %[[#ptr_float:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG: %[[#int_0:]] = OpConstant %[[#int]] 0
; CHECK-DAG: %[[#int_1:]] = OpConstant %[[#int]] 1
; CHECK-DAG: %[[#int_2:]] = OpConstant %[[#int]] 2

define spir_func float @two_deep(ptr %obj) convergent {
entry:
  %tok = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#obj2:]] = OpFunctionParameter %[[#ptr_Outer]]
  ; CHECK: %[[#ptr2:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#obj2]] %[[#int_0]] %[[#int_2]] %[[#int_1]] %[[#int_1]]
  ; CHECK-NOT: OpInBoundsAccessChain
  ; CHECK: OpLoad %[[#float]] %[[#ptr2]]
  %inner = getelementptr inbounds %struct.Outer, ptr %obj, i32 0, i32 0, i32 2
  %vecp = getelementptr inbounds %struct.Inner, ptr %inner, i32 0, i32 1
  %eltp = getelementptr inbounds <4 x float>, ptr %vecp, i32 0, i32 1
  %val = load float, ptr %eltp, align 4
  ret float %val
}

define spir_func float @three_deep(ptr %obj) convergent {
entry:
  %tok = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#obj3:]] = OpFunctionParameter %[[#ptr_Outer]]
  ; CHECK: %[[#ptr3:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#obj3]] %[[#int_0]] %[[#int_2]] %[[#int_1]] %[[#int_1]]
  ; CHECK-NOT: OpInBoundsAccessChain
  ; CHECK: OpLoad %[[#float]] %[[#ptr3]]
  %a = getelementptr inbounds %struct.Outer, ptr %obj, i32 0, i32 0
  %b = getelementptr inbounds [5 x %struct.Inner], ptr %a, i32 0, i32 2
  %c = getelementptr inbounds %struct.Inner, ptr %b, i32 0, i32 1
  %d = getelementptr inbounds <4 x float>, ptr %c, i32 0, i32 1
  %val = load float, ptr %d, align 4
  ret float %val
}

; If the inner GEP has multiple uses, do NOT coalesce: emit two access chains.
define spir_func float @inner_multi_use(ptr %obj) convergent {
entry:
  %tok = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#objm:]] = OpFunctionParameter %[[#ptr_Outer]]
  ; CHECK: %[[#innerp:]] = OpInBoundsAccessChain {{%[0-9]+}} %[[#objm]] %[[#int_0]] %[[#int_2]] %[[#int_1]]
  ; CHECK: %[[#eltp:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#innerp]] %[[#int_1]]
  ; CHECK: OpStore %[[#innerp]]
  %inner = getelementptr inbounds %struct.Outer, ptr %obj, i32 0, i32 0, i32 2, i32 1
  %eltp = getelementptr inbounds <4 x float>, ptr %inner, i32 0, i32 1
  store <4 x float> zeroinitializer, ptr %inner, align 16
  %val = load float, ptr %eltp, align 4
  ret float %val
}

; If any index is dynamic (non-constant), do NOT coalesce.
define spir_func float @dynamic_index(ptr %obj, i32 %i) convergent {
entry:
  %tok = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#objd:]] = OpFunctionParameter %[[#ptr_Outer]]
  ; CHECK: %[[#i:]] = OpFunctionParameter %[[#int]]
  ; CHECK: %[[#dyn:]] = OpInBoundsAccessChain {{%[0-9]+}} %[[#objd]] %[[#int_0]] %[[#i]]
  ; CHECK: OpInBoundsAccessChain %[[#ptr_float]] %[[#dyn]] %[[#int_1]] %[[#int_1]]
  %inner = getelementptr inbounds %struct.Outer, ptr %obj, i32 0, i32 0, i32 %i
  %eltp = getelementptr inbounds %struct.Inner, ptr %inner, i32 0, i32 1, i32 1
  %val = load float, ptr %eltp, align 4
  ret float %val
}

declare token @llvm.experimental.convergence.entry() #1

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
