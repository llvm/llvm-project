; RUN: llc -mtriple=spirv-unknown-unknown -O0 %s -o - | FileCheck %s

; CHECK-DAG:                  OpDecorate %[[#SubgroupLocalInvocationId:]] BuiltIn SubgroupLocalInvocationId
; CHECK-DAG:    %[[#bool:]] = OpTypeBool
; CHECK-DAG:    %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:  %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:                  OpName %[[#main:]] "main"

define void @main() #1 {
  %1 = icmp ne i32 0, 0
  %t1 = call token @llvm.experimental.convergence.entry()
  br i1 %1, label %l1, label %l2

; CHECK:         %[[#main]] = OpFunction
; CHECK:        %[[#cond:]] = OpINotEqual %[[#bool]] %[[#uint_0]] %[[#uint_0]]
; CHECK:                      OpBranchConditional %[[#cond]] %[[#l1_pre:]] %[[#l2_pre:]]

; CHECK-DAG:   %[[#l2_pre]] = OpLabel
; CHECK-NEXT:                 OpBranch %[[#l2_header:]]

; CHECK-DAG:    %[[#l2_header]] = OpLabel
; CHECK-NEXT:                     OpBranchConditional %[[#cond]] %[[#l2_body:]] %[[#l2_end:]]

; CHECK-DAG:   %[[#l2_end]] = OpLabel
; CHECK-NEXT:                 OpBranch %[[#end:]]

; CHECK-DAG:       %[[#end]] = OpLabel
; CHECK-NEXT:                  OpReturn

; CHECK-DAG:   %[[#l2_body]] = OpLabel
; CHECK-NEXT:                  OpBranch %[[#l2_continue:]]

; CHECK-DAG:   %[[#l2_continue]] = OpLabel
; CHECK-NEXT:                      OpBranch %[[#l2_header]]

l1:
  %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
  br i1 %1, label %l1_body, label %l1_end

; CHECK-DAG:   %[[#l1_pre]] = OpLabel
; CHECK-NEXT:                 OpBranch %[[#l1_header:]]

; CHECK-DAG:    %[[#l1_header]] = OpLabel
; CHECK-NEXT:                     OpBranchConditional %[[#cond]] %[[#l1_body:]] %[[#l1_end:]]

; CHECK-DAG:   %[[#l1_end]] = OpLabel
; CHECK-DAG:         %[[#]] = OpLoad %[[#]] %[[#SubgroupLocalInvocationId]]
; CHECK-NEXT:                 OpBranch %[[#end]]

l1_body:
  br label %l1_continue
; CHECK-DAG:   %[[#l1_body]] = OpLabel
; CHECK-NEXT:                  OpBranch %[[#l1_continue:]]

l1_continue:
  br label %l1
; CHECK-DAG:   %[[#l1_continue]] = OpLabel
; CHECK-NEXT:                      OpBranch %[[#l1_header]]

l1_end:
  %call = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %tl1) ]
  br label %end

l2:
  %tl2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
  br i1 %1, label %l2_body, label %l2_end

l2_body:
  br label %l2_continue

l2_continue:
  br label %l2

l2_end:
  br label %end

end:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" convergent }

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.control()
declare token @llvm.experimental.convergence.loop()

declare i32 @__hlsl_wave_get_lane_index() convergent
