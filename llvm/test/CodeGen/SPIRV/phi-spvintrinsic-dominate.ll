; The goal of the test is to check that newly inserted internal (spv)
; intrinsic functions for PHI's operands are inserted at the correct
; positions, and don't break rules of instruction domination and PHI nodes
; grouping at top of basic block.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction
; CHECK: OpBranch
; CHECK: OpLabel
; CHECK: OpPhi
; CHECK: OpPhi
; CHECK: OpPhi

define spir_kernel void @foo(ptr addrspace(1) %_arg1) {
entry:
  br label %l1

l1:
  %sw = phi <4 x double> [ %vec, %l2 ], [ <double 0.0, double 0.0, double 0.0, double poison>, %entry ]
  %in = phi <3 x double> [ %ins, %l2 ], [ zeroinitializer, %entry ]
  %r1 = phi i32 [ %r2, %l2 ], [ 0, %entry ]
  %c1 = icmp ult i32 %r1, 3
  br i1 %c1, label %l2, label %exit

l2:
  %r3 = zext nneg i32 %r1 to i64
  %r4 = getelementptr inbounds double, ptr addrspace(1) %_arg1, i64 %r3
  %r5 = load double, ptr addrspace(1) %r4, align 8
  %ins = insertelement <3 x double> %in, double %r5, i32 %r1
  %exp = shufflevector <3 x double> %ins, <3 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %vec = shufflevector <4 x double> %exp, <4 x double> %sw, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %r2 = add nuw nsw i32 %r1, 1
  br label %l1

exit:
  ret void
}
