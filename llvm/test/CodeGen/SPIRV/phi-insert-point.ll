; The goal of the test is to check that internal intrinsic functions for PHI's
; operand are inserted at the correct positions, and don't break rules of
; instruction domination and PHI nodes grouping at top of basic block.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK: %[[#Foo]] = OpFunction
; CHECK: OpPhi
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpPhi
; CHECK: %[[#Bar]] = OpFunction
; CHECK: OpPhi
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpPhi

%struct = type { i64, i64 }

define spir_kernel void @foo(i64 %arg_val, ptr addrspace(4) byval(%struct) %arg_ptr) {
entry:
  %fl = icmp eq i64 %arg_val, 0
  br i1 %fl, label %ok, label %err

err:
  br label %ok

ok:
  %r1 = phi i64 [ undef, %err ], [ %arg_val, %entry ]
  %r2 = phi i64 [ undef, %err ], [ %arg_val, %entry ]
  %r3 = phi ptr addrspace(4) [ undef, %err ], [ %arg_ptr, %entry ]
  %r4 = phi ptr addrspace(4) [ undef, %err ], [ %arg_ptr, %entry ]
  br label %exit

exit:
  ret void
}

define spir_kernel void @bar(i64 %arg_val, i64 %arg_val_def, ptr addrspace(4) byval(%struct) %arg_ptr, ptr addrspace(4) %arg_ptr_def) {
entry:
  %fl = icmp eq i64 %arg_val, 0
  br i1 %fl, label %ok, label %err

err:
  br label %ok

ok:
  %r1 = phi i64 [ %arg_val_def, %err ], [ %arg_val, %entry ]
  %r2 = phi i64 [ %arg_val_def, %err ], [ %arg_val, %entry ]
  %r3 = phi ptr addrspace(4) [ %arg_ptr_def, %err ], [ %arg_ptr, %entry ]
  %r4 = phi ptr addrspace(4) [ %arg_ptr_def, %err ], [ %arg_ptr, %entry ]
  br label %exit

exit:
  ret void
}
