; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; Normally icmp + select is optimized to select_cc, when this happens the
; DAGLegalizer never sees the select and doesn't have a chance to leaglize it.
;
; In order to avoid the select_cc optimization, this test case calculates the
; condition for the select in a separate basic block.

; FUNC-LABEL: {{^}}select:
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.X
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.X
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XY
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XY
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XYZW
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XYZW
define amdgpu_kernel void @select (ptr addrspace(1) %i32out, ptr addrspace(1) %f32out,
                     ptr addrspace(1) %v2i32out, ptr addrspace(1) %v2f32out,
                     ptr addrspace(1) %v4i32out, ptr addrspace(1) %v4f32out,
                     i32 %cond) {
entry:
  br label %for
body:
  %inc = add i32 %i, 1
  %br_cmp.i = icmp eq i1 %br_cmp, 0
  br label %for
for:
  %i = phi i32 [ %inc, %body], [ 0, %entry ]
  %br_cmp = phi i1 [ %br_cmp.i, %body ], [ 0, %entry ]
  %0 = icmp eq i32 %cond, %i
  %1 = select i1 %br_cmp, i32 2, i32 3
  %2 = select i1 %br_cmp, float 2.0 , float 5.0
  %3 = select i1 %br_cmp, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 4, i32 5>
  %4 = select i1 %br_cmp, <2 x float> <float 2.0, float 3.0>, <2 x float> <float 4.0, float 5.0>
  %5 = select i1 %br_cmp, <4 x i32> <i32 2 , i32 3, i32 4, i32 5>, <4 x i32> <i32 6, i32 7, i32 8, i32 9>
  %6 = select i1 %br_cmp, <4 x float> <float 2.0, float 3.0, float 4.0, float 5.0>, <4 x float> <float 6.0, float 7.0, float 8.0, float 9.0>
  br i1 %0, label %body, label %done

done:
  store i32 %1, ptr addrspace(1) %i32out
  store float %2, ptr addrspace(1) %f32out
  store <2 x i32> %3, ptr addrspace(1) %v2i32out
  store <2 x float> %4, ptr addrspace(1) %v2f32out
  store <4 x i32> %5, ptr addrspace(1) %v4i32out
  store <4 x float> %6, ptr addrspace(1) %v4f32out
  ret void
}
