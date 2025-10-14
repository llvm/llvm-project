; Example input for update_test_checks (taken from test/Transforms/SLPVectorizer/extractlements-gathered-first-node.ll)
; RUN: %if x86-registered-target %{ opt -S --passes=slp-vectorizer -slp-threshold=-99999 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s %}
; RUN: %if aarch64-registered-target %{ opt -S --passes=slp-vectorizer -slp-threshold=-99999 -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s %}

define void @test() {
bb:
  %0 = extractelement <4 x i32> zeroinitializer, i32 0
  %1 = extractelement <2 x i32> zeroinitializer, i32 0
  %icmp = icmp ult i32 %0, %1
  ret void
}
