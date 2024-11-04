; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction
; CHECK: OpBranchConditional %[[#]] %[[#if_then:]] %[[#if_end:]]
; CHECK: %[[#if_end]] = OpLabel
; CHECK: %[[#Var:]] = OpPhi
; CHECK: OpSwitch %[[#Var]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]]
; CHECK-COUNT-11: OpLabel
; CHECK-NOT: OpBranch
; CHECK: OpReturn
; CHECK: %[[#if_then]] = OpLabel
; CHECK: OpBranch %[[#if_end]]
; CHECK-NEXT: OpFunctionEnd

define spir_func void @foo(i64 noundef %addr, i64 noundef %as) {
entry:
  %src = inttoptr i64 %as to ptr addrspace(4)
  %val = load i8, ptr addrspace(4) %src
  %cmp = icmp sgt i8 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %src, i64 1
  %cond = load i8, ptr addrspace(4) %add.ptr
  br label %if.end

if.end:
  %swval = phi i8 [ %cond, %if.then ], [ %val, %entry ]
  switch i8 %swval, label %sw.default [
    i8 -127, label %sw.epilog
    i8 -126, label %sw.bb3
    i8 -125, label %sw.bb4
    i8 -111, label %sw.bb5
    i8 -110, label %sw.bb6
    i8 -109, label %sw.bb7
    i8 -15, label %sw.bb8
    i8 -14, label %sw.bb8
    i8 -13, label %sw.bb8
    i8 -124, label %sw.bb9
    i8 -95, label %sw.bb10
    i8 -123, label %sw.bb11
  ]

sw.bb3:
  br label %sw.epilog

sw.bb4:
  br label %sw.epilog

sw.bb5:
  br label %sw.epilog

sw.bb6:
  br label %sw.epilog

sw.bb7:
  br label %sw.epilog

sw.bb8:
  br label %sw.epilog

sw.bb9:
  br label %sw.epilog

sw.bb10:
  br label %sw.epilog

sw.bb11:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %exit

exit:
  ret void
}
