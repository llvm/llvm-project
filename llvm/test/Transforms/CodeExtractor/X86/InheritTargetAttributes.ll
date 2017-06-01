; RUN: opt < %s -partial-inliner | llc -filetype=null
; RUN: opt < %s -partial-inliner -S | FileCheck %s
; This testcase checks to see if CodeExtractor properly inherits
;   target specific attributes for the extracted function. This can
;   cause certain instructions that depend on the attributes to not
;   be lowered. Like in this test where we try to 'select' the blendvps
;   intrinsic on x86 that requires the +sse4.1 target feature.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) #0

; Function Attrs: nounwind uwtable
define <4 x float> @inlinedFunc(i1, <4 x float>, <4 x float>, <4 x float>) #1 {
entry:
  br i1 %0, label %if.then, label %return
if.then:
; Target intrinsic that requires sse4.1
  %target.call = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %1, <4 x float> %2, <4 x float> %3)
  br label %return
return:             ; preds = %entry
  %retval = phi <4 x float> [ zeroinitializer, %entry ], [ %target.call, %if.then ]
  ret <4 x float> %retval
}

; Function Attrs: nounwind uwtable
define <4 x float> @dummyCaller(i1, <4 x float>, <4 x float>, <4 x float>) #1 {
entry:
  %val = call <4 x float> @inlinedFunc(i1 %0, <4 x float> %1, <4 x float> %2, <4 x float> %3)
  ret <4 x float> %val
}


attributes #0 = { nounwind readnone }
attributes #1 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+sse4.1" }

; CHECK: define {{.*}} @inlinedFunc.1_if.then{{.*}} [[COUNT1:#[0-9]+]]
; CHECK: [[COUNT1]] = { {{.*}} "target-cpu"="x86-64" "target-features"="+sse4.1" }
