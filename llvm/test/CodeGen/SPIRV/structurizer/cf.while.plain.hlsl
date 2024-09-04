// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

int foo() { return true; }

[numthreads(1, 1, 1)]
void main() {
  int val = 0;
  int i = 0;

    //////////////////////////
    //// Basic while loop ////
    //////////////////////////

// CHECK:      OpBranch %while_check
// CHECK-NEXT: %while_check = OpLabel

// CHECK-NEXT: [[i:%[0-9]+]] = OpLoad %int %i
// CHECK-NEXT: [[i_lt_10:%[0-9]+]] = OpSLessThan %bool [[i]] %int_10
// CHECK-NEXT: OpLoopMerge %while_merge %while_continue None
// CHECK-NEXT: OpBranchConditional [[i_lt_10]] %while_body %while_merge
  while (i < 10) {
// CHECK-NEXT: %while_body = OpLabel
// CHECK-NEXT: [[i1:%[0-9]+]] = OpLoad %int %i
// CHECK-NEXT: OpStore %val [[i1]]
      val = i;
// CHECK-NEXT: OpBranch %while_continue
// CHECK-NEXT: %while_continue = OpLabel
// CHECK-NEXT: OpBranch %while_check
  }
// CHECK-NEXT: %while_merge = OpLabel



    //////////////////////////
    ////  infinite loop   ////
    //////////////////////////

// CHECK-NEXT: OpBranch %while_check_0
// CHECK-NEXT: %while_check_0 = OpLabel
// CHECK-NEXT: OpLoopMerge %while_merge_0 %while_continue_0 None
// CHECK-NEXT: OpBranchConditional %true %while_body_0 %while_merge_0
  while (true) {
// CHECK-NEXT: %while_body_0 = OpLabel
// CHECK-NEXT: OpStore %val %int_0
      val = 0;
// CHECK-NEXT: OpBranch %while_continue_0
// CHECK-NEXT: %while_continue_0 = OpLabel
// CHECK-NEXT: OpBranch %while_check_0
  }
// CHECK-NEXT: %while_merge_0 = OpLabel
// CHECK-NEXT: OpBranch %while_check_1



    //////////////////////////
    ////    Null Body     ////
    //////////////////////////

// CHECK-NEXT: %while_check_1 = OpLabel
// CHECK-NEXT: [[val1:%[0-9]+]] = OpLoad %int %val
// CHECK-NEXT: [[val_lt_20:%[0-9]+]] = OpSLessThan %bool [[val1]] %int_20
// CHECK-NEXT: OpLoopMerge %while_merge_1 %while_continue_1 None
// CHECK-NEXT: OpBranchConditional [[val_lt_20]] %while_body_1 %while_merge_1
  while (val < 20)
// CHECK-NEXT: %while_body_1 = OpLabel
// CHECK-NEXT: OpBranch %while_continue_1
// CHECK-NEXT: %while_continue_1 = OpLabel
// CHECK-NEXT: OpBranch %while_check_1
    ;
// CHECK-NEXT: %while_merge_1 = OpLabel
// CHECK-NEXT: OpBranch %while_check_2



    ////////////////////////////////////////////////////////////////
    //// Condition variable has VarDecl                         ////
    //// foo() returns an integer which must be cast to boolean ////
    ////////////////////////////////////////////////////////////////

// CHECK-NEXT: %while_check_2 = OpLabel
// CHECK-NEXT: [[foo:%[0-9]+]] = OpFunctionCall %int %foo
// CHECK-NEXT: OpStore %a [[foo]]
// CHECK-NEXT: [[a:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: [[is_a_true:%[0-9]+]] = OpINotEqual %bool [[a]] %int_0
// CHECK-NEXT: OpLoopMerge %while_merge_2 %while_continue_2 None
// CHECK-NEXT: OpBranchConditional [[is_a_true]] %while_body_2 %while_merge_2
  while (int a = foo()) {
// CHECK-NEXT: %while_body_2 = OpLabel
// CHECK-NEXT: [[a1:%[0-9]+]] = OpLoad %int %a
// CHECK-NEXT: OpStore %val [[a1]]
    val = a;
// CHECK-NEXT: OpBranch %while_continue_2
// CHECK-NEXT: %while_continue_2 = OpLabel
// CHECK-NEXT: OpBranch %while_check_2
  }
// CHECK-NEXT: %while_merge_2 = OpLabel


// CHECK-NEXT: OpReturn
// CHECK-NEXT: OpFunctionEnd
}
