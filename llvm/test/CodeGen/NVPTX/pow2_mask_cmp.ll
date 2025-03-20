; RUN: llc -mtriple=nvptx64 -verify-machineinstrs < %s | FileCheck %s
; RUN: %if ptxas %{ llc -mtriple=nvptx64 -verify-machineinstrs < %s | %ptxas-verify %}

; Tests the following pattern:
; (X & 8) != 0 --> (X & 8) >> 3

; This produces incorrect code in general when boolean false is
; represented as a negative one. There is however a special
; case when the type has a bitsize of 1, for which the false
; value will be identical regardless of the boolean representation.
; Check that the optimization triggers in this case.

; CHECK-LABEL: @pow2_mask_cmp
; CHECK: bfe.u32 {{%r[0-9]+}}, {{%r[0-9]+}}, 3, 1
define i32 @pow2_mask_cmp(i32 %x) {
  %a = and i32 %x, 8
  %cmp = icmp ne i32 %a, 0
  %r = zext i1 %cmp to i32
  ret i32 %r
}
