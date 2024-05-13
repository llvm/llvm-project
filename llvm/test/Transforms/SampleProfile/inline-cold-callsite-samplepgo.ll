; For SamplePGO, if -profile-sample-accurate is specified, cold callsite
; heuristics should be honored if the caller has no profile.

; RUN: opt -S %s -passes='sample-profile,cgscc(inline)' -sample-profile-file=%S/Inputs/inline.prof -inline-cold-callsite-threshold=0 | FileCheck %s
; RUN: opt -S %s -passes='sample-profile,cgscc(inline)' -sample-profile-file=%S/Inputs/inline.prof -profile-sample-accurate -inline-cold-callsite-threshold=0 | FileCheck %s --check-prefix ACCURATE

declare void @extern()
define void @callee() #1 {
  call void @extern()
  ret void
}

define void @caller(i32 %y1) #1 {
; CHECK-LABEL: @caller
; CHECK-NOT: call void @callee
; ACCURATE-LABEL: @caller
; ACCURATE: call void @callee
  call void @callee()
  ret void
}

define void @caller_accurate(i32 %y1) #0 {
; CHECK-LABEL: @caller_accurate
; CHECK: call void @callee
; ACCURATE-LABEL: @caller_accurate
; ACCURATE: call void @callee
  call void @callee()
  ret void
}

attributes #0 = { "profile-sample-accurate" "use-sample-profile" }
attributes #1 = { "use-sample-profile" }
