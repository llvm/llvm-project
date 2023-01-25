; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; There was an infinite loop in DAGCombiner from a target build_vector
; combine and a generic insert_vector_elt combine.

; GCN-LABEL: {{^}}combine_loop:
; GCN: flat_load_short_d16_hi
; GCN: flat_store_short
define amdgpu_kernel void @combine_loop(ptr %arg) #0 {
bb:
  br label %bb1

bb1:
  %tmp = phi <2 x i16> [ <i16 15360, i16 15360>, %bb ], [ %tmp5, %bb1 ]
  %tmp2 = phi half [ 0xH0000, %bb ], [ %tmp8, %bb1 ]
  %tmp3 = load volatile half, ptr null, align 536870912
  %tmp4 = bitcast half %tmp3 to i16
  %tmp5 = insertelement <2 x i16> <i16 0, i16 undef>, i16 %tmp4, i32 1
  store volatile half %tmp2, ptr %arg, align 2
  %tmp7 = bitcast <2 x i16> %tmp to <2 x half>
  %tmp8 = extractelement <2 x half> %tmp7, i32 0
  br label %bb1
}

attributes #0 = { nounwind }
