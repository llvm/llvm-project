; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -print-after=si-annotate-control-flow %s -o /dev/null 2>&1 | FileCheck %s

target datalayout = "n32"

; CHECK-LABEL: @switch_unreachable_default

define amdgpu_kernel void @switch_unreachable_default(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) #0 {
centry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  switch i32 %tid, label %sw.default [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
  ]

sw.bb0:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.default:
  unreachable

sw.epilog:
  %ptr = phi ptr addrspace(1) [%in0, %sw.bb0], [%in1, %sw.bb1]
  br label %sw.while

; The loop below is necessary to preserve the effect of the
; unreachable default on divergence analysis in the presence of other
; optimizations. The loop consists of a single block where the loop
; exit is divergent because it depends on the divergent phi at the
; start of the block. The checks below ensure that the loop exit is
; handled correctly as divergent. But the data-flow within the block
; is sensitive to optimizations; so we just ensure that the relevant
; operations in the block body are indeed in the same block.

; CHECK: [[PHI:%[a-zA-Z0-9._]+]]  = phi i64
; CHECK-NOT: {{ br }}
; CHECK: load i8
; CHECK-NOT: {{ br }}
; CHECK: [[ICMP:%[a-zA-Z0-9._]+]] = icmp eq
; CHECK: [[IF:%[a-zA-Z0-9._]+]]   = call i64 @llvm.amdgcn.if.break.i64(i1 [[ICMP]], i64 [[PHI]])
; CHECK: [[LOOP:%[a-zA-Z0-9._]+]] = call i1 @llvm.amdgcn.loop.i64(i64 [[IF]])
; CHECK: br i1 [[LOOP]]

sw.while:
  %p = phi ptr addrspace(1) [ %ptr, %sw.epilog ], [ %incdec.ptr, %sw.while ]
  %count = phi i32 [ 0, %sw.epilog ], [ %count.inc, %sw.while ]
  %char = load i8, ptr addrspace(1) %p, align 1
  %tobool = icmp eq i8 %char, 0
  %incdec.ptr = getelementptr inbounds i8, ptr addrspace(1) %p, i64 1
  %count.inc = add i32 %count, 1
  br i1 %tobool, label %sw.exit, label %sw.while

sw.exit:
  %tid64 = zext i32 %tid to i64
  %gep_out = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid64
  store i32 %count, ptr addrspace(1) %gep_out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { convergent noinline optnone }
