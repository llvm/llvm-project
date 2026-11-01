; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-LABEL: llvm.func @with_elision_policy()
; CHECK-SAME: sample_profile_suffix_elision_policy = "selected"
define void @with_elision_policy() #0 {
  ret void
}

; CHECK-LABEL: llvm.func @without_elision_policy()
; CHECK-NOT: sample_profile_suffix_elision_policy
define void @without_elision_policy() {
  ret void
}

attributes #0 = { "sample-profile-suffix-elision-policy"="selected" }
