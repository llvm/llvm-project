; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck  %s
; XXX: Merge this test into vselect.ll once SI supports 64-bit select.

; CHECK-LABEL: {{^}}test_select_v4i64:
; Make sure the vectors aren't being stored on the stack.  We know they are
; being stored on the stack if the shaders uses at leat 10 registers.
; CHECK-NOT: {{\**}} MOV T{{[0-9][0-9]}}.X
define amdgpu_kernel void @test_select_v4i64(<4 x i64> addrspace(1)* %out, <4 x i32> %c) {
entry:
       %cmp = icmp ne  <4 x i32> %c, <i32 0, i32 0, i32 0, i32 0>
       %result = select <4 x i1> %cmp, <4 x i64> <i64 0, i64 1, i64 2, i64 3>, <4 x i64> <i64 4, i64 5, i64 6, i64 7>
       store <4 x i64> %result, <4 x i64> addrspace(1)* %out
       ret void
}

