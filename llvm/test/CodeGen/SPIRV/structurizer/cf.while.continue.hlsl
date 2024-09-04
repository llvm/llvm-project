// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return true; }

[numthreads(1, 1, 1)]
void main() {
  int val = 0;
  int i = 0;

  
// CHECK:      OpBranch %while_check
// CHECK:      %while_check = OpLabel
// CHECK:      [[i_lt_10:%[0-9]+]] = OpSLessThan %bool {{%[0-9]+}} %int_10
// CHECK-NEXT: OpLoopMerge %while_merge %while_continue None
// CHECK-NEXT: OpBranchConditional [[i_lt_10]] %while_body %while_merge
  while (i < 10) {
// CHECK-NEXT: %while_body = OpLabel
    val = i;
// CHECK:      [[val_gt_5:%[0-9]+]] = OpSGreaterThan %bool {{%[0-9]+}} %int_5
// CHECK-NEXT: OpSelectionMerge %if_merge None
// CHECK-NEXT: OpBranchConditional [[val_gt_5]] %if_true %if_merge
    if (val > 5) {
// CHECK-NEXT: %if_true = OpLabel
// CHECK-NEXT: OpBranch %while_continue
      continue;
    }
// CHECK-NEXT: %if_merge = OpLabel

// CHECK:      [[val_gt_6:%[0-9]+]] = OpSGreaterThan %bool {{%[0-9]+}} %int_6
// CHECK-NEXT: OpSelectionMerge %if_merge_0 None
// CHECK-NEXT: OpBranchConditional [[val_gt_6]] %if_true_0 %if_merge_0
    if (val > 6) {
// CHECK-NEXT: %if_true_0 = OpLabel
// CHECK-NEXT: OpBranch %while_continue
      {{continue;}}
      val++;       // No SPIR-V should be emitted for this statement.
      continue;    // No SPIR-V should be emitted for this statement.
      while(true); // No SPIR-V should be emitted for this statement.
      --i;         // No SPIR-V should be emitted for this statement.
    }
// CHECK-NEXT: %if_merge_0 = OpLabel

// CHECK-NEXT: OpBranch %while_continue
// CHECK-NEXT: %while_continue = OpLabel
// CHECK-NEXT: OpBranch %while_check
  }

// CHECK-NEXT: %while_merge = OpLabel



  //////////////////////////////////////////////////////////////////////////////////////
  // Nested while loops with continue statements                                      //
  // Each continue statement should branch to the corresponding loop's continue block //
  //////////////////////////////////////////////////////////////////////////////////////

// CHECK-NEXT: OpBranch %while_check_0
// CHECK-NEXT: %while_check_0 = OpLabel
// CHECK-NEXT: OpLoopMerge %while_merge_1 %while_continue_1 None
// CHECK-NEXT: OpBranchConditional %true %while_body_0 %while_merge_1
  while (true) {
// CHECK-NEXT: %while_body_0 = OpLabel
    i++;

// CHECK:      OpBranch %while_check_1
// CHECK-NEXT: %while_check_1 = OpLabel
// CHECK:      [[i_lt_20:%[0-9]+]] = OpSLessThan %bool {{%[0-9]+}} %int_20
// CHECK-NEXT: OpLoopMerge %while_merge_0 %while_continue_0 None
// CHECK-NEXT: OpBranchConditional [[i_lt_20]] %while_body_1 %while_merge_0
    while(i<20) {
// CHECK-NEXT: %while_body_1 = OpLabel
      val = i;
// CHECK:      OpBranch %while_continue_0
      continue;
// CHECK-NEXT: %while_continue_0 = OpLabel
// CHECK-NEXT: OpBranch %while_check_1
    }
// CHECK-NEXT: %while_merge_0 = OpLabel
    --i;
// CHECK:      OpBranch %while_continue_1
    continue;
    continue;  // No SPIR-V should be emitted for this statement.

// CHECK-NEXT: %while_continue_1 = OpLabel
// CHECK-NEXT: OpBranch %while_check_0
  }
// CHECK-NEXT: %while_merge_1 = OpLabel

}
