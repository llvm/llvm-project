; RUN: opt -S -passes=dse < %s | FileCheck %s

; DSE must not remove stores to addrspace(5) (per-lane private scratch) when
; the function contains convergent memory(none) calls (e.g. ballot/bpermute).
; Such calls are invisible to MemorySSA but observe private scratch cross-lane.

declare void @shfl(ptr addrspace(5)) convergent readnone
declare i32 @llvm.amdgcn.ballot.i32(i1) convergent memory(none)
declare void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) noalias writeonly, ptr noalias readonly, i64, i1 immarg)

; Warp reduction: store intermediate result, shuffle reads it cross-lane,
; then overwrite. DSE must not remove the first store.
define void @warp_reduction_intermediate_store(ptr addrspace(5) %warpVal) #0 {
; CHECK-LABEL: @warp_reduction_intermediate_store(
; CHECK: store float 1.000000e+00, ptr addrspace(5) %warpVal
;
entry:
  store float 1.0, ptr addrspace(5) %warpVal, align 4
  call void @shfl(ptr addrspace(5) %warpVal)
  store float 2.0, ptr addrspace(5) %warpVal, align 4
  ret void
}

; Conditional kill: zero-init is overwritten by memcpy only on active lanes.
; Inactive lanes still hold the zero-init when the ballot observes all lanes.
; DSE must not remove the zero-init store.
define i32 @fitbounds_conditional_kill(i1 %active, ptr %src) #0 {
; CHECK-LABEL: @fitbounds_conditional_kill(
; CHECK: store i32 0, ptr addrspace(5) %node
;
entry:
  %node = alloca i32, align 4, addrspace(5)
  store i32 0, ptr addrspace(5) %node, align 4
  br i1 %active, label %if.then, label %if.end

if.then:
  call void @llvm.memcpy.p5.p0.i64(ptr addrspace(5) %node, ptr %src, i64 4, i1 false)
  %loaded = load i32, ptr addrspace(5) %node, align 4
  %pred = icmp ne i32 %loaded, 0
  br label %if.end

if.end:
  %p = phi i1 [ false, %entry ], [ %pred, %if.then ]
  %v = call i32 @llvm.amdgcn.ballot.i32(i1 %p)
  ret i32 %v
}

attributes #0 = { convergent }
