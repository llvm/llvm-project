; RUN: llc < %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #0

define i32 @offload_rpc_output_s0___outermost_par_for_output_s0_y_line_chunk_chunk(<64 x i32> %0, ptr %linearized) #1 {
entry:
  br label %"for linearized.s0.x.x"

"for linearized.s0.x.x":                          ; preds = %"for linearized.s0.x.x", %entry
  %1 = add <64 x i32> %0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %2 = call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %1)
  store <32 x i32> %2, ptr %linearized, align 128
  br label %"for linearized.s0.x.x"
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { "target-features"="+hvx-length128b,+long-calls,+hvxv62" }
