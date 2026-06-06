; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -disable-hsdr < %s | FileCheck %s

; Check that the parsing succeeded.
; CHECK: r{{[0-9]+:[0-9]+}} = xor(r{{[0-9]+:[0-9]+}},r{{[0-9]+:[0-9]+}})
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

@window_size = global i32 65536, align 4
@prev = external global [0 x i16], align 8
@block_start = common global i32 0, align 4
@prev_length = common global i32 0, align 4
@strstart = common global i32 0, align 4
@match_start = common global i32 0, align 4
@max_chain_length = common global i32 0, align 4
@good_match = common global i32 0, align 4

define void @fill_window() #0 {
entry:
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret void

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %polly.loopiv36 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv36, 4
  %p_arrayidx4 = getelementptr [0 x i16], ptr @prev, i32 0, i32 %polly.loopiv36
  %_p_vec_full = load <4 x i16>, ptr %p_arrayidx4, align 2
  %cmp1p_vicmp = icmp slt <4 x i16> %_p_vec_full, zeroinitializer
  %subp_vec = xor <4 x i16> %_p_vec_full, <i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %sel1p_vsel = select <4 x i1> %cmp1p_vicmp, <4 x i16> %subp_vec, <4 x i16> zeroinitializer
  store <4 x i16> %sel1p_vsel, ptr %p_arrayidx4, align 2
  %0 = icmp slt i32 %polly.next_loopiv, 32768
  br i1 %0, label %polly.loop_body, label %polly.loop_after
}

attributes #0 = { nounwind "fp-contract-model"="standard" "frame-pointer"="non-leaf" "realign-stack" "relocation-model"="static" "ssp-buffers-size"="8" }
