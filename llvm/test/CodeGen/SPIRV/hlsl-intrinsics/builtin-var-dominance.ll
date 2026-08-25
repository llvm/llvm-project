; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

;; Verify correct translation when the same builtin is called from multiple
;; non-entry blocks. The fix is validated by -verify-machineinstrs that fails
;; if the OpVariable's VReg definition does not dominate all its uses in MIR.

; CHECK-DAG: OpDecorate %[[#VarID:]] BuiltIn LocalInvocationId
; CHECK-DAG: %[[#VarID]] = OpVariable %[[#]] Input

define internal spir_func void @main_inner(<3 x i32> noundef %ID) {
entry:
  ret void
}

define void @main.1() #0 {
entry:
  %cmp = icmp sgt i32 1, 0
  br i1 %cmp, label %then, label %else

then:
  %0 = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %1 = insertelement <3 x i32> poison, i32 %0, i64 0
  %2 = call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
  %3 = insertelement <3 x i32> %1, i32 %2, i64 1
  %4 = call i32 @llvm.spv.thread.id.in.group.i32(i32 2)
  %5 = insertelement <3 x i32> %3, i32 %4, i64 2
  call void @main_inner(<3 x i32> %5)
  br label %exit

else:
  %6 = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %7 = insertelement <3 x i32> poison, i32 %6, i64 0
  %8 = call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
  %9 = insertelement <3 x i32> %7, i32 %8, i64 1
  %10 = call i32 @llvm.spv.thread.id.in.group.i32(i32 2)
  %11 = insertelement <3 x i32> %9, i32 %10, i64 2
  call void @main_inner(<3 x i32> %11)
  br label %exit

exit:
  ret void
}

declare i32 @llvm.spv.thread.id.in.group.i32(i32) #1

attributes #0 = { norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #1 = { nounwind willreturn memory(none) }
