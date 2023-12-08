; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability GroupNonUniformBallot
; CHECK-SPIRV: OpDecorate %[[#]] BuiltIn SubgroupGtMask

;; kernel void test_mask(global uint4 *out)
;; {
;;   *out = get_sub_group_gt_mask();
;; }

define dso_local spir_kernel void @test_mask(<4 x i32> addrspace(1)* nocapture noundef writeonly %out) local_unnamed_addr {
entry:
  %call = tail call spir_func <4 x i32> @_Z21get_sub_group_gt_maskv()
  store <4 x i32> %call, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

declare spir_func <4 x i32> @_Z21get_sub_group_gt_maskv() local_unnamed_addr
