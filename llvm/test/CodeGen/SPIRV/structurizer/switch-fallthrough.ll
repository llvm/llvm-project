; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-compute"

define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = tail call token @llvm.experimental.convergence.entry()
  %2 = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  switch i32 %2, label %sw1 [
    i32 0, label %sw0
    i32 2, label %sw0
  ]

sw0:
  %5 = call spir_func i1 @llvm.spv.wave.any(i1 false) [ "convergencectrl"(token %0) ]
  br label %exit

sw1:
  %6 = call spir_func i1 @llvm.spv.wave.any(i1 false) [ "convergencectrl"(token %0) ]
  br label %exit

exit:
  %.sink = phi i1 [ %5, %sw0 ], [ %6, %sw1 ]
  %conv.i = zext i1 %.sink to i32
  ret i32 %conv.i
}

define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #0 [ "convergencectrl"(token %0) ]
  ret void
}

declare i1 @llvm.spv.wave.any(i1) #0
declare i32 @llvm.spv.thread.id.i32(i32) #2
declare token @llvm.experimental.convergence.entry() #0

attributes #0 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "frame-pointer"="all" "hlsl.numthreads"="4,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(none) }
