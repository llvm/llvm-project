; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --implicit-check-not=OpSwitch
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; These CFGs model the conditional branch chains which are roughly equivalent
; to switch statements but without using switch instructions. In particular, the
; break-only chain used to make an outer selection merge branch into a nested
; selection's merge block.

; CHECK-DAG: OpName %[[FALLTHROUGH:[0-9]+]] "fallthrough"
; CHECK-DAG: OpName %[[BREAKS:[0-9]+]] "breaks"
; CHECK-DAG: OpName %[[MIXED:[0-9]+]] "mixed"

; CHECK: %[[FALLTHROUGH]] = OpFunction
; CHECK-COUNT-3: OpSelectionMerge
; CHECK: OpFunctionEnd
define i32 @fallthrough(i32 %value) {
entry:
  br label %sw.dispatch

sw.dispatch:
  %case0 = icmp eq i32 %value, 0
  br i1 %case0, label %sw.bb0, label %sw.next0

sw.bb0:
  br label %sw.next0

sw.next0:
  %fellthrough0 = phi i1 [ true, %sw.bb0 ], [ false, %sw.dispatch ]
  %case1 = icmp eq i32 %value, 1
  %take1 = or i1 %fellthrough0, %case1
  br i1 %take1, label %sw.bb1, label %sw.next1

sw.bb1:
  br label %sw.next1

sw.next1:
  %fellthrough1 = phi i1 [ true, %sw.bb1 ], [ false, %sw.next0 ]
  %not.case0 = icmp ne i32 %value, 0
  %not.case1 = icmp ne i32 %value, 1
  %take.default = and i1 %not.case0, %not.case1
  %take2 = or i1 %fellthrough1, %take.default
  br i1 %take2, label %sw.default, label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  %result = phi i32 [ 7, %sw.default ], [ 0, %sw.next1 ]
  ret i32 %result
}

; CHECK: %[[BREAKS]] = OpFunction
; CHECK-NOT: OpSelectionMerge
; CHECK: OpSelectionMerge %[[BREAK_EPILOG:[0-9]+]] None
; CHECK-NEXT: OpBranchConditional %{{[0-9]+}} %[[BREAK_BB0:[0-9]+]] %[[BREAK_NEXT0:[0-9]+]]
; CHECK: %[[BREAK_NEXT0]] = OpLabel
; CHECK-NOT: OpSelectionMerge
; CHECK: OpSelectionMerge %[[BREAK_NEXT0_EXIT:[0-9]+]] None
; CHECK-NEXT: OpBranchConditional %{{[0-9]+}} %[[BREAK_NEXT0_EXIT]] %[[BREAK_NEXT1:[0-9]+]]
; CHECK: %[[BREAK_NEXT1]] = OpLabel
; CHECK-NOT: OpSelectionMerge
; CHECK: OpSelectionMerge %[[BREAK_NEXT1_EXIT:[0-9]+]] None
; CHECK-NEXT: OpBranchConditional %{{[0-9]+}} %[[BREAK_NEXT1_EXIT]] %[[BREAK_CLEANUP:[0-9]+]]
; CHECK: %[[BREAK_CLEANUP]] = OpLabel
; CHECK: OpBranch %[[BREAK_NEXT1_EXIT]]
; CHECK: %[[BREAK_NEXT1_EXIT]] = OpLabel
; CHECK: OpBranchConditional %{{[0-9]+}} %[[BREAK_DEFAULT:[0-9]+]] %[[BREAK_NEXT0_EXIT]]
; CHECK: %[[BREAK_DEFAULT]] = OpLabel
; CHECK: OpBranch %[[BREAK_NEXT0_EXIT]]
; CHECK: %[[BREAK_NEXT0_EXIT]] = OpLabel
; CHECK: OpBranchConditional %{{[0-9]+}} %[[BREAK_BB1:[0-9]+]] %[[BREAK_EPILOG]]
; CHECK: %[[BREAK_BB1]] = OpLabel
; CHECK: OpBranch %[[BREAK_EPILOG]]
; CHECK: %[[BREAK_BB0]] = OpLabel
; CHECK: OpBranch %[[BREAK_EPILOG]]
; CHECK: %[[BREAK_EPILOG]] = OpLabel
; CHECK-NOT: OpSelectionMerge
; CHECK: OpFunctionEnd
define i32 @breaks(i32 %value) {
entry:
  br label %sw.dispatch

sw.dispatch:
  %case0 = icmp eq i32 %value, 0
  br i1 %case0, label %sw.bb0, label %sw.next0

sw.bb0:
  br label %sw.epilog

sw.next0:
  %case1 = icmp eq i32 %value, 1
  br i1 %case1, label %sw.bb1, label %sw.next1

sw.bb1:
  br label %sw.epilog

sw.next1:
  %not.case0 = icmp ne i32 %value, 0
  %not.case1 = icmp ne i32 %value, 1
  %take.default = and i1 %not.case0, %not.case1
  br i1 %take.default, label %sw.default, label %sw.cleanup

sw.default:
  br label %sw.epilog

sw.cleanup:
  br label %sw.epilog

sw.epilog:
  %result = phi i32 [ 1, %sw.bb0 ], [ 2, %sw.bb1 ],
                    [ 4, %sw.default ], [ 0, %sw.cleanup ]
  ret i32 %result
}

; CHECK: %[[MIXED]] = OpFunction
; CHECK-COUNT-4: OpSelectionMerge
; CHECK: OpFunctionEnd
define i32 @mixed(i32 %value) {
entry:
  br label %sw.dispatch

sw.dispatch:
  %case0 = icmp eq i32 %value, 0
  br i1 %case0, label %sw.bb0, label %sw.next0

sw.bb0:
  br label %sw.next0

sw.next0:
  %fellthrough0 = phi i1 [ true, %sw.bb0 ], [ false, %sw.dispatch ]
  %case1 = icmp eq i32 %value, 1
  %take1 = or i1 %fellthrough0, %case1
  br i1 %take1, label %sw.bb1, label %sw.next1

sw.bb1:
  br label %sw.epilog

sw.next1:
  %case2 = icmp eq i32 %value, 2
  br i1 %case2, label %sw.bb2, label %sw.next2

sw.bb2:
  br label %sw.next2

sw.next2:
  %fellthrough2 = phi i1 [ true, %sw.bb2 ], [ false, %sw.next1 ]
  %not.case0 = icmp ne i32 %value, 0
  %not.case1 = icmp ne i32 %value, 1
  %not.case2 = icmp ne i32 %value, 2
  %not.case01 = and i1 %not.case0, %not.case1
  %take.default = and i1 %not.case01, %not.case2
  %take3 = or i1 %fellthrough2, %take.default
  br i1 %take3, label %sw.default, label %sw.cleanup

sw.default:
  br label %sw.epilog

sw.cleanup:
  br label %sw.epilog

sw.epilog:
  %result = phi i32 [ 3, %sw.bb1 ], [ 12, %sw.default ],
                    [ 0, %sw.cleanup ]
  ret i32 %result
}
