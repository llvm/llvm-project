; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT: %_s.sroa.343.0.ph5161118.ph.merge = phi i32 [ %_s.sroa.343.0.ph5161118.ph.final_reload, %polly.exiting ], [ %_s.sroa.343.0.ph516.lcssa2357, %for.cond.981.region_exiting ]

; CHECK-LABEL: for.cond.981:
; CHECK-NEXT:  %_s.sroa.343.0.ph5161118 = phi i32 [ undef, %for.cond ], [ %_s.sroa.343.0.ph5161118.ph.merge, %polly.merge_new_and_old ]

; CHECK-LABEL: polly.exiting:
; CHECK-NEXT: %_s.sroa.343.0.ph5161118.ph.final_reload = load i32, i32* %_s.sroa.343.0.ph5161118.s2a

; Function Attrs: nounwind uwtable
define void @lzmaDecode() #0 {
entry:
  br label %for.cond.outer.outer.outer

for.cond:                                         ; preds = %for.cond.outer.outer.outer
  switch i32 undef, label %cleanup.1072 [
    i32 23, label %for.cond.981
    i32 4, label %_LZMA_C_RDBD
    i32 19, label %sw.bb.956
    i32 26, label %saveStateAndReturn
  ]

_LZMA_C_RDBD:                                     ; preds = %for.cond
  ret void

sw.bb.956:                                        ; preds = %for.cond
  %_s.sroa.294.0.ph519.lcssa2388 = phi i32 [ undef, %for.cond ]
  %_s.sroa.343.0.ph516.lcssa2357 = phi i32 [ undef, %for.cond ]
  %cmp958 = icmp eq i32 %_s.sroa.294.0.ph519.lcssa2388, 0
  br i1 %cmp958, label %if.then.960, label %if.else.969

if.then.960:                                      ; preds = %sw.bb.956
  br label %for.cond.981

if.else.969:                                      ; preds = %sw.bb.956
  br label %for.cond.981

for.cond.981:                                     ; preds = %if.else.969, %if.then.960, %for.cond
  %_s.sroa.343.0.ph5161118 = phi i32 [ %_s.sroa.343.0.ph516.lcssa2357, %if.then.960 ], [ %_s.sroa.343.0.ph516.lcssa2357, %if.else.969 ], [ undef, %for.cond ]
  ret void

for.cond.outer.outer.outer:                       ; preds = %entry
  br label %for.cond

saveStateAndReturn:                               ; preds = %for.cond
  ret void

cleanup.1072:                                     ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
