;; Emit the runtime hook even if there are no counter increments.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes=instrprof -S | FileCheck %s -check-prefixes=ALL,DARWIN
; RUN: opt < %s -mtriple=x86_64-linux-unknown -passes=instrprof -S | FileCheck %s -check-prefixes=ALL,LINUX
; RUN: opt < %s -mtriple=powerpc64-ibm-aix-xcoff -passes=instrprof -S | FileCheck %s -check-prefixes=ALL,DARWIN
; ALL-NOT: @__profc
; ALL-NOT: @__profd
; DARWIN: @__llvm_profile_runtime
; LINUX-NOT: @__llvm_profile_runtime

define void @foo() {
  ret void
}
