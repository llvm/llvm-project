; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind
; CHECK: .func kernelgen_memcpy
define ptx_device void @kernelgen_memcpy(i8* nocapture %dst) #0 {
entry:
  br i1 undef, label %for.end, label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %scevgep9 = getelementptr i8, i8* %dst, i64 %index
  %scevgep910 = bitcast i8* %scevgep9 to <4 x i8>*
  store <4 x i8> undef, <4 x i8>* %scevgep910, align 1
  %index.next = add i64 %index, 4
  %0 = icmp eq i64 undef, %index.next
  br i1 %0, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  br i1 undef, label %for.end, label %for.body.preheader1

for.body.preheader1:                              ; preds = %middle.block
  %scevgep2 = getelementptr i8, i8* %dst, i64 0
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader1
  %lsr.iv3 = phi i8* [ %scevgep2, %for.body.preheader1 ], [ %scevgep4, %for.body ]
  store i8 undef, i8* %lsr.iv3, align 1
  %scevgep4 = getelementptr i8, i8* %lsr.iv3, i64 1
  br label %for.body

for.end:                                          ; preds = %middle.block, %entry
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
