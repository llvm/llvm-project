; REQUIRES: asserts
; RUN: llc < %s -mtriple=thumbv7-apple-ios -disable-ifcvt-diamond -stats 2>&1 | FileCheck %s
; Evaluate the two vld1.8 instructions in separate MBB's,
; instead of stalling on one and conditionally overwriting its result.
;
; Update: After if-conversion the two vld1.8 instructions are in the same MBB
; again. So we disable this if-conversion to eliminate its influence to this
; test.

; CHECK-NOT: Number of pipeline stalls
define <16 x i8> @multiselect(i32 %avail, ptr %foo, ptr %bar) {
entry:
  %vld1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr %foo, i32 1)
  %vld2 = call <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr %bar, i32 1)
  %and = and i32 %avail, 3
  %tobool = icmp eq i32 %and, 0
  %retv = select i1 %tobool, <16 x i8> %vld1, <16 x i8> %vld2
  ret <16 x i8> %retv
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8.p0(ptr , i32 )
