; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; The loop header switch has a single merged exit and a single latch.
; CHECK: OpLoopMerge %[[#exit:]] %[[#latch:]] None
; CHECK: OpSelectionMerge %[[#merge:]] None

; The switch keeps exactly three cases. This confirms the redundant case that
; duplicated the default exit is removed not split into a fourth case / extra
; exit block. The end-of-line anchor ensures no additional case is appended.
; CHECK: OpSwitch %[[#]] %[[#merge]] 2 %[[#]] 0 %[[#]] 1 %[[#]]{{$}}

; The selection merge routes to the single loop exit or back to the latch.
; CHECK: OpBranchConditional %[[#]] %[[#exit]] %[[#latch]]

define void @main() #0 {
entry:
  %t0 = tail call token @llvm.experimental.convergence.entry()
  br label %for.cond

for.cond:
  %I = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %tl = tail call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t0) ]
  switch i32 %I, label %exit [
    i32 4, label %exit
    i32 0, label %c0
    i32 1, label %c1
    i32 2, label %c2
  ]

c0:
  %v0 = call i32 @get(i32 0) [ "convergencectrl"(token %tl) ]
  br label %latch

c1:
  %v1 = call i32 @get(i32 1) [ "convergencectrl"(token %tl) ]
  br label %latch

c2:
  %v2 = call i32 @get(i32 2) [ "convergencectrl"(token %tl) ]
  br label %latch

latch:
  %acc = phi i32 [ %v0, %c0 ], [ %v1, %c1 ], [ %v2, %c2 ]
  call void @put(i32 %I, i32 %acc) [ "convergencectrl"(token %tl) ]
  %inc = add nuw nsw i32 %I, 1
  br label %for.cond

exit:
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1
declare i32 @get(i32) #2
declare void @put(i32, i32) #2

attributes #0 = { convergent noinline norecurse nounwind "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent }
