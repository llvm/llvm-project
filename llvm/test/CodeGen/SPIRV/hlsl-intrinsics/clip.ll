; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV15
; RUN: llc -verify-machineinstrs -spirv-ext=+SPV_EXT_demote_to_helper_invocation -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV16
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}


; Make sure lowering is correctly generating spirv code.

; CHECK-DAG:   %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#void:]]  = OpTypeVoid
; CHECK-DAG:   %[[#bool:]]  = OpTypeBool
; CHECK-DAG:   %[[#v4bool:]]  = OpTypeVector %[[#bool]] 4
; CHECK-DAG:   %[[#v4float:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG:   %[[#fzero:]] = OpConstant %[[#float]] 0
; CHECK-DAG:   %[[#v4fzero:]] = OpConstantNull %[[#v4float]]
; SPIRV16-DAG: %[[#vecfuncopptr:]] = OpTypePointer Function %[[#v4float]]
; SPIRV16-DAG: %[[#funcopptr:]] = OpTypePointer Function %[[#float]]

define void @test_scalar(float noundef %Buf) {
entry:
; CHECK-LABEL: ; -- Begin function test_scalar
; SPIRV16:     %[[#param:]] = OpVariable %[[#funcopptr]] Function
; SPIRV16:     %[[#load:]] = OpLoad %[[#float]] %[[#param]] Aligned 4
; SPIRV15:     %[[#load:]] = OpFunctionParameter %[[#float]]
; CHECK:       %[[#cmplt:]] = OpFOrdLessThan %[[#bool]] %[[#load]] %[[#fzero]]
; CHECK:       OpBranchConditional %[[#cmplt]] %[[#truel:]] %[[#endl:]]
; CHECK:       %[[#truel]] = OpLabel
; SPIRV15:     OpKill
; SPIRV16-NO:  OpKill
; SPIRV15-NO:  OpBranch %[[#endl]]
; SPIRV16:     OpDemoteToHelperInvocation
; SPIRV16:     OpBranch %[[#endl]]
; CHECK:       %[[#endl]] = OpLabel
  %Buf.addr = alloca float, align 4
  store float %Buf, ptr %Buf.addr, align 4
  %1 = load float, ptr %Buf.addr, align 4
  %2 = fcmp olt float %1, 0.000000e+00
  br i1 %2, label %lt0, label %end

lt0:                                              ; preds = %entry
  call void @llvm.spv.clip()
  br label %end

end:                                              ; preds = %lt0, %entry
  ret void
}
declare void @llvm.spv.clip()

define void @test_vector(<4 x float> noundef %Buf) {
entry:
; CHECK-LABEL: ; -- Begin function test_vector
; SPIRV16:     %[[#param:]] = OpVariable %[[#vecfuncopptr]] Function
; SPIRV16:     %[[#loadvec:]] = OpLoad %[[#v4float]] %[[#param]] Aligned 16
; SPIRV15:     %[[#loadvec:]] = OpFunctionParameter %[[#v4float]]
; CHECK:       %[[#cmplt:]] = OpFOrdLessThan %[[#v4bool]] %[[#loadvec]] %[[#v4fzero]]
; CHECK:       %[[#opany:]] = OpAny %[[#bool]] %[[#cmplt]]
; CHECK:       OpBranchConditional %[[#opany]]  %[[#truel:]] %[[#endl:]]
; CHECK:       %[[#truel]] = OpLabel
; SPIRV15:     OpKill
; SPIRV16-NO:  OpKill
; SPIRV15-NO:  OpBranch %[[#endl]]
; SPIRV16:     OpDemoteToHelperInvocation
; SPIRV16:     OpBranch %[[#endl]]
; CHECK:       %[[#endl]] = OpLabel
  %Buf.addr = alloca <4 x float>, align 16
  store <4 x float> %Buf, ptr %Buf.addr, align 16
  %1 = load <4 x float>, ptr %Buf.addr, align 16
  %2 = fcmp olt <4 x float> %1, zeroinitializer
  %3 = call i1 @llvm.spv.any.v4i1(<4 x i1> %2)
  br i1 %3, label %lt0, label %end

lt0:                                              ; preds = %entry
  call void @llvm.spv.clip()
  br label %end

end:                                              ; preds = %lt0, %entry
  ret void
}
