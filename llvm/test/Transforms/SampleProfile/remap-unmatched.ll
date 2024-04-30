; This test should not crash.
; RUN: opt %s -passes=sample-profile -sample-profile-file=%S/Inputs/remap.prof -sample-profile-remapping-file=%S/Inputs/remap.map

define void @foo() #0 {
  ret void
}

attributes #0 = { "use-sample-profile" }
