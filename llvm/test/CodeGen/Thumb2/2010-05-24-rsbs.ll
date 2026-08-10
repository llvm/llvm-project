; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s
; Radar 8017376: Missing 's' suffix for t2RSBS instructions.
; CHECK: rsbs

define i64 @test(i64 %x) nounwind readnone {
entry:
  %0 = sub nsw i64 1, %x                          ; <i64> [#uses=1]
  ret i64 %0
}
