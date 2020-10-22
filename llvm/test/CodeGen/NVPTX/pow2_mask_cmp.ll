; RUN: llc -march=nvptx -verify-machineinstrs < %s | FileCheck %s

; Tests the following pattern:
; (X & 8) != 0 --> (X & 8) >> 3

; This produces incorrect code when boolean false is represented
; as a negative one, and this test checks that the transform is
; not triggered.

; CHECK-LABEL: @pow2_mask_cmp
; CHECK: and.b32 [[AND:%r[0-9]+]], %r{{[0-9]+}}, 8
; CHECK: setp.ne.s32 [[SETP:%p[0-9+]]], [[AND]], 0
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, [[SETP]]
define i32 @pow2_mask_cmp(i32 %x) {
  %a = and i32 %x, 8
  %cmp = icmp ne i32 %a, 0
  %r = zext i1 %cmp to i32
  ret i32 %r
}
