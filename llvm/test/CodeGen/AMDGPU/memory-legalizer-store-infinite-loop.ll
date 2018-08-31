; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Effectively, check that the compile finishes; in the case
; of an infinite loop, llc toggles between merging 2 ST4s
; ( MergeConsecutiveStores() ) and breaking the resulting ST8
; apart ( LegalizeStoreOps() ).

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"

; GCN-LABEL: {{^}}_Z6brokenPd:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}
define amdgpu_kernel void @_Z6brokenPd(double* %arg) {
bb:
  %tmp = alloca double, align 8, addrspace(5)
  %tmp1 = alloca double, align 8, addrspace(5)
  %tmp2 = load double, double* %arg, align 8
  br i1 1, label %bb6, label %bb4

bb3:                                             ; No predecessors!
  br label %bb4

bb4:                                             ; preds = %bb3, %bb
  %tmp5 = phi double addrspace(5)* [ %tmp1, %bb3 ], [ %tmp, %bb ]
  store double %tmp2, double addrspace(5)* %tmp5, align 8
  br label %bb6

bb6:                                             ; preds = %bb4, %bb
  %tmp7 = phi double [ 0x7FF8123000000000, %bb4 ], [ 0x7FF8000000000000, %bb ]
  store double %tmp7, double* %arg, align 8
  ret void
}
