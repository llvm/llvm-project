; RUN: llc -mtriple="arm64-apple-ios" < %s | FileCheck %s
;
; Check that the dead register definition pass is considering implicit defs.
; When rematerializing through truncates, the coalescer may produce instructions
; with dead defs, but live implicit-defs of subregs:
; E.g. dead %x1 = MOVi64imm 2, implicit-def %w1; %x1:GPR64, %w1:GPR32
; These instructions are live, and their definitions should not be rewritten.
;
; <rdar://problem/16492408>

define void @testcase() {
; CHECK: testcase:
; CHECK-NOT: orr xzr, xzr, #0x2

bb1:
  %tmp1 = tail call float @ceilf(float 2.000000e+00)
  %tmp2 = fptoui float %tmp1 to i64
  br i1 undef, label %bb2, label %bb3

bb2:
  tail call void @foo()
  br label %bb3

bb3:
  %tmp3 = trunc i64 %tmp2 to i32
  tail call void @bar(i32 %tmp3)
  ret void
}

declare void @foo()
declare void @bar(i32)
declare float @ceilf(float) nounwind readnone
