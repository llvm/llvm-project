; RUN: llc -mtriple=spirv-unknown-unknown -O0 %s -o - | FileCheck %s

; CHECK-DAG:  [[bool:%[0-9]+]] = OpTypeBool
; CHECK-DAG:  [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG:  [[uint_0:%[0-9]+]] = OpConstant [[uint]] 0

define i32 @main(i32 noundef %0) #1 {
  %2 = icmp ne i32 %0, 0
  br i1 %2, label %l1, label %l2

; CHECK: [[param_0:%[0-9]+]] = OpFunctionParameter [[uint]]
; CHECK:    [[cond:%[0-9]+]] = OpINotEqual [[bool]] [[param_0]] [[uint_0]]
; CHECK:                       OpBranchConditional [[cond]] [[l1_pre:%[0-9]+]] [[l2_pre:%[0-9]+]]

; CHECK-DAG:   [[l2_pre]] = OpLabel
; CHECK-NEXT:               OpBranch [[l2_header:%[0-9]+]]

; CHECK-DAG:   [[l1_pre]] = OpLabel
; CHECK-NEXT:               OpBranch [[l1_header:%[0-9]+]]

l1:
  br i1 %2, label %l1_body, label %l1_end
; CHECK-DAG:    [[l1_header]] = OpLabel
; CHECK-NEXT:                   OpBranchConditional [[cond]] [[l1_body:%[0-9]+]] [[l1_end:%[0-9]+]]

l1_body:
  br label %l1_continue
; CHECK-DAG:   [[l1_body]] = OpLabel
; CHECK-NEXT:                OpBranch [[l1_continue:%[0-9]+]]

l1_continue:
  br label %l1
; CHECK-DAG:   [[l1_continue]] = OpLabel
; CHECK-NEXT:                    OpBranch [[l1_header]]

l1_end:
  br label %end
; CHECK-DAG:   [[l1_end]] = OpLabel
; CHECK-NEXT:                OpBranch [[end:%[0-9]+]]

l2:
  br i1 %2, label %l2_body, label %l2_end
; CHECK-DAG:    [[l2_header]] = OpLabel
; CHECK-NEXT:                   OpBranchConditional [[cond]] [[l2_body:%[0-9]+]] [[l2_end:%[0-9]+]]

l2_body:
  br label %l2_continue
; CHECK-DAG:   [[l2_body]] = OpLabel
; CHECK-NEXT:                OpBranch [[l2_continue:%[0-9]+]]

l2_continue:
  br label %l2
; CHECK-DAG:   [[l2_continue]] = OpLabel
; CHECK-NEXT:                    OpBranch [[l2_header]]

l2_end:
  br label %end
; CHECK-DAG:   [[l2_end]] = OpLabel
; CHECK-NEXT:                OpBranch [[end:%[0-9]+]]

end:
  ret i32 1
; CHECK-DAG:       [[end]] = OpLabel
; CHECK-NEXT:                OpReturn
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" convergent }
