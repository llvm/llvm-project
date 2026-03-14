; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13
;
; This tescase origininates from the BB-vectorizer output.

define void @fun() {
  %1 = zext <3 x i1> zeroinitializer to <3 x i32>
  %2 = extractelement <3 x i32> %1, i32 2
  store i32 %2, ptr undef, align 8
  unreachable
}
