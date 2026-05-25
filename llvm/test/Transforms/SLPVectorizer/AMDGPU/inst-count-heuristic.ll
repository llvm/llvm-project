; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes=slp-vectorizer %s \
; RUN:   | FileCheck -check-prefix=DEFAULT %s

; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes=slp-vectorizer \
; RUN:     -slp-inst-count-check=true %s \
; RUN:   | FileCheck -check-prefix=COUNTCHECK %s

; The -slp-inst-count-check heuristic rejects size-2 vector trees whose
; lowered vector instruction count exceeds the scalar count. With the
; heuristic disabled (default) the rotating chain of five i32 phis below
; stays scalar; with the heuristic enabled SLP packs four of those phis
; into <2 x i32> phis. The resulting code then has one wider carrier
; move in the loop body, which is unwanted on AMDGPU gfx94x and gfx950.

; DEFAULT-LABEL: @rotating_phi(
; DEFAULT:       phi i32 [ 1, %ph ]

; COUNTCHECK-LABEL: @rotating_phi(
; COUNTCHECK:       phi <2 x i32> [ <i32 0, i32 1>, %ph ]

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @rotating_phi(ptr addrspace(1) %p, i64 %niter, i1 %skip, i32 %seed) {
entry:
  %init6 = load i32, ptr addrspace(1) %p, align 4
  br i1 %skip, label %ph, label %exit

ph:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid64 = zext i32 %tid to i64
  br label %loop

loop:
  %iv  = phi i64 [ %tid64, %ph ], [ %iv.next, %loop ]
  %s6  = phi i32 [ 1, %ph ], [ %s8,  %loop ]
  %s8  = phi i32 [ 0, %ph ], [ %s10, %loop ]
  %s10 = phi i32 [ 0, %ph ], [ %s12, %loop ]
  %s12 = phi i32 [ 0, %ph ], [ %s14, %loop ]
  %s14 = phi i32 [ 0, %ph ], [ %s6,  %loop ]
  %iv.next = add i64 %iv, 1
  %cond    = icmp ult i64 %iv, %niter
  br i1 %cond, label %loop, label %exit

exit:
  %r6 = phi i32 [ %init6, %entry ], [ %s8,   %loop ]
  %r8 = phi i32 [ 0,      %entry ], [ %s10,  %loop ]
  %ro = phi i32 [ 0,      %entry ], [ %seed, %loop ]
  store i32 %ro, ptr addrspace(1) %p, align 4
  %p1 = getelementptr i8, ptr addrspace(1) %p, i64 4
  store i32 %r6, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr i8, ptr addrspace(1) %p, i64 8
  store i32 %r8, ptr addrspace(1) %p2, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
