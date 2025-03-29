; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s
; REQUIRES: asserts
; Check for successful compilation.

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

@input_buf = internal unnamed_addr constant [256 x i16] [i16 0, i16 0, i16 0, i16 1280, i16 2560, i16 4864, i16 7168, i16 9472, i16 11776, i16 12672, i16 13568, i16 14080, i16 15360, i16 15360, i16 15360, i16 15360, i16 15360, i16 15104, i16 14848, i16 14592, i16 14336, i16 14080, i16 14080, i16 13952, i16 13824, i16 13696, i16 13568, i16 13440, i16 13312, i16 13184, i16 13056, i16 12928, i16 12800, i16 12800, i16 12800, i16 12800, i16 12800, i16 12672, i16 12544, i16 12544, i16 12544, i16 12544, i16 12672, i16 12800, i16 12800, i16 12928, i16 13056, i16 13184, i16 13312, i16 13440, i16 13568, i16 13696, i16 13824, i16 14208, i16 14592, i16 14976, i16 15104, i16 15360, i16 15616, i16 15872, i16 16128, i16 16512, i16 16896, i16 17152, i16 17408, i16 17536, i16 17664, i16 17792, i16 17920, i16 18304, i16 18688, i16 19072, i16 19456, i16 19712, i16 19968, i16 20224, i16 20480, i16 20608, i16 20864, i16 20992, i16 21248, i16 21248, i16 21248, i16 21248, i16 21248, i16 21248, i16 21376, i16 21504, i16 21760, i16 21760, i16 21632, i16 21504, i16 21504, i16 21632, i16 21632, i16 21504, i16 21504, i16 21376, i16 21248, i16 21120, i16 20992, i16 20992, i16 20864, i16 20736, i16 20736, i16 20736, i16 20480, i16 20352, i16 20224, i16 20224, i16 20224, i16 20224, i16 20352, i16 20352, i16 20480, i16 20352, i16 20352, i16 20352, i16 20352, i16 20224, i16 20224, i16 20224, i16 20096, i16 20096, i16 19968, i16 19840, i16 19712, i16 19584, i16 19456, i16 19584, i16 19584, i16 19456, i16 19456, i16 19328, i16 19328, i16 19456, i16 19456, i16 19328, i16 19328, i16 19200, i16 19200, i16 19200, i16 19072, i16 19072, i16 18944, i16 18816, i16 18688, i16 18560, i16 18432, i16 18304, i16 18304, i16 18176, i16 18176, i16 18176, i16 18304, i16 18304, i16 18432, i16 18560, i16 18432, i16 18176, i16 17920, i16 17920, i16 17792, i16 17792, i16 17664, i16 17664, i16 17536, i16 17536, i16 17408, i16 17408, i16 17280, i16 17280, i16 17280, i16 17152, i16 17152, i16 17152, i16 17152, i16 17024, i16 17024, i16 16896, i16 16896, i16 16896, i16 16768, i16 16768, i16 16640, i16 16640, i16 16512, i16 16512, i16 16384, i16 16256, i16 16128, i16 16000, i16 15872, i16 15744, i16 15616, i16 15488, i16 15360, i16 15488, i16 15360, i16 15232, i16 15360, i16 15232, i16 15104, i16 14976, i16 14336, i16 14336, i16 14592, i16 14464, i16 13824, i16 13824, i16 13568, i16 13568, i16 13440, i16 13312, i16 13184, i16 13056, i16 13056, i16 13056, i16 12928, i16 12800, i16 12672, i16 12672, i16 12544, i16 12416, i16 12288, i16 12160, i16 11904, i16 11776, i16 11571, i16 11520, i16 11392, i16 11136, i16 10905, i16 10752, i16 10624, i16 10444, i16 10240, i16 9984, i16 9728, i16 9472, i16 9216, i16 8960, i16 8704, i16 8448, i16 8192, i16 7936, i16 7680, i16 7424, i16 7168, i16 6400, i16 5632, i16 4864, i16 3584, i16 1536, i16 0, i16 0], align 8

; Function Attrs: nounwind
define i32 @t_run_test() #0 {
entry:
  %WaterLeveldB_out = alloca i16, align 2
  br label %polly.stmt.for.body

for.body8:                                        ; preds = %for.body8, %polly.loop_exit.loopexit
  %i.120 = phi i32 [ 0, %polly.loop_exit.loopexit ], [ %inc11.24, %for.body8 ]
  %call = call i32 @fxpBitAllocation(i32 0, i32 0, i32 256, i32 %conv9, ptr %WaterLeveldB_out, i32 0, i32 1920, i32 %i.120) #2
  %inc11.24 = add i32 %i.120, 25
  %exitcond.24 = icmp eq i32 %inc11.24, 500
  br i1 %exitcond.24, label %for.end12, label %for.body8

for.end12:                                        ; preds = %for.body8
  ret i32 0

polly.loop_exit.loopexit:                         ; preds = %polly.stmt.for.body
  %WaterLeveldB.1p_vsel.lcssa = phi <4 x i16> [ %WaterLeveldB.1p_vsel, %polly.stmt.for.body ]
  %_low_half = shufflevector <4 x i16> %WaterLeveldB.1p_vsel.lcssa, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %_high_half = shufflevector <4 x i16> %WaterLeveldB.1p_vsel.lcssa, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %0 = icmp sgt <2 x i16> %_low_half, %_high_half
  %1 = select <2 x i1> %0, <2 x i16> %_low_half, <2 x i16> %_high_half
  %2 = extractelement <2 x i16> %1, i32 0
  %3 = extractelement <2 x i16> %1, i32 1
  %4 = icmp sgt i16 %2, %3
  %5 = select i1 %4, i16 %2, i16 %3
  %conv9 = sext i16 %5 to i32
  br label %for.body8

polly.stmt.for.body:                              ; preds = %entry, %polly.stmt.for.body
  %WaterLeveldB.1p_vsel35 = phi <4 x i16> [ <i16 -32768, i16 -32768, i16 -32768, i16 -32768>, %entry ], [ %WaterLeveldB.1p_vsel, %polly.stmt.for.body ]
  %scevgep.phi = phi ptr [ @input_buf, %entry ], [ %scevgep.inc, %polly.stmt.for.body ]
  %polly.indvar = phi i32 [ 0, %entry ], [ %polly.indvar_next, %polly.stmt.for.body ]
  %_p_vec_full = load <4 x i16>, ptr %scevgep.phi, align 8
  %cmp2p_vicmp = icmp sgt <4 x i16> %_p_vec_full, %WaterLeveldB.1p_vsel35
  %WaterLeveldB.1p_vsel = select <4 x i1> %cmp2p_vicmp, <4 x i16> %_p_vec_full, <4 x i16> %WaterLeveldB.1p_vsel35
  %polly.indvar_next = add nsw i32 %polly.indvar, 4
  %polly.loop_cond = icmp slt i32 %polly.indvar, 252
  %scevgep.inc = getelementptr i16, ptr %scevgep.phi, i32 4
  br i1 %polly.loop_cond, label %polly.stmt.for.body, label %polly.loop_exit.loopexit
}

declare i32 @fxpBitAllocation(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"QuIC LLVM Hexagon Clang version 3.1"}
