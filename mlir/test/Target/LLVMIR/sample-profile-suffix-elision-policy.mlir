// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @with_elision_policy() #[[ATTRS_WITH:.*]] {
llvm.func @with_elision_policy() attributes {sample_profile_suffix_elision_policy = "selected"} {
  llvm.return
}

// CHECK: define void @without_elision_policy() {
// CHECK-NOT: "sample-profile-suffix-elision-policy"
llvm.func @without_elision_policy() {
  llvm.return
}

// CHECK: attributes #[[ATTRS_WITH]] = { "sample-profile-suffix-elision-policy"="selected" }
