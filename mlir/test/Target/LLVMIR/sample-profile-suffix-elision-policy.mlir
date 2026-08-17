// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @with_elision_policy() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "sample-profile-suffix-elision-policy"="selected" }
llvm.func @with_elision_policy() attributes {sample_profile_suffix_elision_policy = "selected"} {
  llvm.return
}
