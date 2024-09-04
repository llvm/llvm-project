// RUN: %if spirv-tools %{ %clang --driver-mode=dxc -T cs_6_0 -fspv-target-env=vulkan1.3 %s -spirv | spirv-as --preserve-numeric-ids - -o - | spirv-val %}

[numthreads(1, 1, 1)]
void main() {
  int val=0, i=0, j=0, k=0;

// CHECK:      OpBranch %while_check
// CHECK-NEXT: %while_check = OpLabel
// CHECK-NEXT: [[i0:%[0-9]+]] = OpLoad %int %i
// CHECK-NEXT: [[i_lt_10:%[0-9]+]] = OpSLessThan %bool [[i0]] %int_10
// CHECK-NEXT: OpLoopMerge %while_merge_1 %while_continue_1 DontUnroll
// CHECK-NEXT: OpBranchConditional [[i_lt_10]] %while_body %while_merge_1
  while (i < 10) {
// CHECK-NEXT: %while_body = OpLabel
// CHECK-NEXT: [[val1:%[0-9]+]] = OpLoad %int %val
// CHECK-NEXT: [[i1:%[0-9]+]] = OpLoad %int %i
// CHECK-NEXT: [[val_plus_i:%[0-9]+]] = OpIAdd %int [[val1]] [[i1]]
// CHECK-NEXT: OpStore %val [[val_plus_i]]
// CHECK-NEXT: OpBranch %while_check_0
    val = val + i;
// CHECK-NEXT: %while_check_0 = OpLabel
// CHECK-NEXT: [[j0:%[0-9]+]] = OpLoad %int %j
// CHECK-NEXT: [[j_lt_20:%[0-9]+]] = OpSLessThan %bool [[j0]] %int_20
// CHECK-NEXT: OpLoopMerge %while_merge_0 %while_continue_0 Unroll
// CHECK-NEXT: OpBranchConditional [[j_lt_20]] %while_body_0 %while_merge_0
    while (j < 20) {
// CHECK-NEXT: %while_body_0 = OpLabel
// CHECK-NEXT: OpBranch %while_check_1

// CHECK-NEXT: %while_check_1 = OpLabel
// CHECK-NEXT: [[k0:%[0-9]+]] = OpLoad %int %k
// CHECK-NEXT: [[k_lt_30:%[0-9]+]] = OpSLessThan %bool [[k0]] %int_30
// CHECK-NEXT: OpLoopMerge %while_merge %while_continue DontUnroll
// CHECK-NEXT: OpBranchConditional [[k_lt_30]] %while_body_1 %while_merge
      while (k < 30) {
// CHECK-NEXT: %while_body_1 = OpLabel
// CHECK-NEXT: [[val2:%[0-9]+]] = OpLoad %int %val
// CHECK-NEXT: [[k2:%[0-9]+]] = OpLoad %int %k
// CHECK-NEXT: [[val_plus_k:%[0-9]+]] = OpIAdd %int [[val2]] [[k2]]
// CHECK-NEXT: OpStore %val [[val_plus_k]]
        val = val + k;
// CHECK-NEXT: [[k3:%[0-9]+]] = OpLoad %int %k
// CHECK-NEXT: [[k_plus_1:%[0-9]+]] = OpIAdd %int [[k3]] %int_1
// CHECK-NEXT: OpStore %k [[k_plus_1]]
        ++k;
// CHECK-NEXT: OpBranch %while_continue
// CHECK-NEXT: %while_continue = OpLabel
// CHECK-NEXT: OpBranch %while_check_1
      }
// CHECK-NEXT: %while_merge = OpLabel

// CHECK-NEXT: [[val3:%[0-9]+]] = OpLoad %int %val
// CHECK-NEXT: [[val_mult_2:%[0-9]+]] = OpIMul %int [[val3]] %int_2
// CHECK-NEXT: OpStore %val [[val_mult_2]]
      val = val * 2;
// CHECK-NEXT: [[j1:%[0-9]+]] = OpLoad %int %j
// CHECK-NEXT: [[j_plus_1:%[0-9]+]] = OpIAdd %int [[j1]] %int_1
// CHECK-NEXT: OpStore %j [[j_plus_1]]
      ++j;
// CHECK-NEXT: OpBranch %while_continue_0
// CHECK-NEXT: %while_continue_0 = OpLabel
// CHECK-NEXT: OpBranch %while_check_0
    }
// CHECK-NEXT: %while_merge_0 = OpLabel

// CHECK-NEXT: [[i2:%[0-9]+]] = OpLoad %int %i
// CHECK-NEXT: [[i_plus_1:%[0-9]+]] = OpIAdd %int [[i2]] %int_1
// CHECK-NEXT: OpStore %i [[i_plus_1]]
    ++i;
// CHECK-NEXT: OpBranch %while_continue_1
// CHECK-NEXT: %while_continue_1 = OpLabel
// CHECK-NEXT: OpBranch %while_check
  }
// CHECK-NEXT: %while_merge_1 = OpLabel


// CHECK-NEXT: OpReturn
// CHECK-NEXT: OpFunctionEnd
}
