; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -O2 < %s -pipeliner-experimental-cg=true | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -O3 < %s -pipeliner-experimental-cg=true | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s -pipeliner-experimental-cg=true -early-live-intervals -verify-machineinstrs | FileCheck %s
;
; Check that we pipeline a vectorized dot product in a single packet.
;
; CHECK: {
; CHECK: += mpyi
; CHECK: += mpyi
; CHECK: memd
; CHECK: memd
; CHECK: }      :endloop0

@a = common global [5000 x i32] zeroinitializer, align 8
@b = common global [5000 x i32] zeroinitializer, align 8

define i32 @vecMultGlobal() {
entry:
  br label %polly.loop_body

polly.loop_after:
  %0 = extractelement <2 x i32> %addp_vec, i32 0
  %1 = extractelement <2 x i32> %addp_vec, i32 1
  %add_sum = add i32 %0, %1
  ret i32 %add_sum

polly.loop_body:
  %polly.loopiv13 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %reduction.012 = phi <2 x i32> [ zeroinitializer, %entry ], [ %addp_vec, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv13, 2
  %p_arrayidx1 = getelementptr [5000 x i32], ptr @b, i32 0, i32 %polly.loopiv13
  %p_arrayidx = getelementptr [5000 x i32], ptr @a, i32 0, i32 %polly.loopiv13
  %_p_vec_full = load <2 x i32>, ptr %p_arrayidx1, align 8
  %_p_vec_full8 = load <2 x i32>, ptr %p_arrayidx, align 8
  %mulp_vec = mul <2 x i32> %_p_vec_full8, %_p_vec_full
  %addp_vec = add <2 x i32> %mulp_vec, %reduction.012
  %2 = icmp slt i32 %polly.next_loopiv, 5000
  br i1 %2, label %polly.loop_body, label %polly.loop_after
}
