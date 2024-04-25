; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --translator-compatibility-mode %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --translator-compatibility-mode %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[Long:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Array:.*]] = OpTypeArray %[[Long]] %[[#]]
; CHECK-DAG: %[[Struct:.*]] = OpTypeStruct %[[Array]]
; CHECK-DAG: %[[StructPtr:.*]] = OpTypePointer Function %[[Struct]]
; CHECK-DAG: %[[CharPtr:.*]] = OpTypePointer Function %[[Char]]

; CHECK: %[[Branch1:.*]] = OpLabel
; CHECK: %[[Res1:.*]] = OpVariable %[[StructPtr]] Function
; CHECK: OpBranchConditional %[[#]] %[[#]] %[[Branch2:.*]]
; CHECK: %[[Res2:.*]] = OpInBoundsPtrAccessChain %[[CharPtr]] %[[#]] %[[#]]
; CHECK: %[[Res2Casted:.*]] = OpBitcast %[[StructPtr]] %[[Res2]]
; CHECK: OpBranchConditional %[[#]] %[[#]] %[[BranchSelect:.*]]
; CHECK: %[[SelectRes:.*]] = OpSelect %[[CharPtr]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[SelectResCasted:.*]] = OpBitcast %[[StructPtr]] %[[SelectRes]]
; CHECK: OpLabel
; CHECK: OpPhi %[[StructPtr]] %[[Res1]] %[[Branch1]] %[[Res2Casted]] %[[Branch2]] %[[SelectResCasted]] %[[BranchSelect]]

%struct = type { %array }
%array = type { [1 x i64] }
%array3 = type { [3 x i32] }

define spir_kernel void @foo(ptr addrspace(1) noundef align 1 %arg1, ptr noundef byval(%struct) align 8 %arg2, i1 noundef zeroext %expected) {
entry:
  %agg = alloca %array3, align 8
  %r0 = load i64, ptr %arg2, align 8
  %add.ptr = getelementptr inbounds i8, ptr %agg, i64 12
  %r1 = load i32, ptr %agg, align 4
  %tobool0 = icmp slt i32 %r1, 0
  br i1 %tobool0, label %exit, label %sw1

sw1:                            ; preds = %entry
  %incdec1 = getelementptr inbounds i8, ptr %agg, i64 4
  %r2 = load i32, ptr %incdec1, align 4
  %tobool1 = icmp slt i32 %r2, 0
  br i1 %tobool1, label %exit, label %sw2

sw2:                            ; preds = %sw1
  %incdec2 = getelementptr inbounds i8, ptr %agg, i64 8
  %r3 = load i32, ptr %incdec2, align 4
  %tobool2 = icmp slt i32 %r3, 0
  %spec.select = select i1 %tobool2, ptr %incdec2, ptr %add.ptr
  br label %exit

exit: ; preds = %sw2, %sw1, %entry
  %retval.0 = phi ptr [ %agg, %entry ], [ %incdec1, %sw1 ], [ %spec.select, %sw2 ]
  %add.ptr.i = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %r0
  %r4 = icmp eq ptr %retval.0, %add.ptr
  %cmp = xor i1 %r4, %expected
  %frombool6.i = zext i1 %cmp to i8
  store i8 %frombool6.i, ptr addrspace(1) %add.ptr.i, align 1
  %r5 = icmp eq ptr %add.ptr, %retval.0
  ret void
}
