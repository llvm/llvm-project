; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mattr=+altivec -mattr=-power8-vector -mattr=-vsx < %s | FileCheck %s
; XFAIL: *

define dso_local void @test(float %0) local_unnamed_addr {
entry:
  %1 = fptosi float %0 to i32
  store i32 %1, i32* undef, align 4
  ret void
}
