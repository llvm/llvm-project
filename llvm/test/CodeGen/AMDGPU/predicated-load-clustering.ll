; REQUIRES: asserts
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32 -verify-machineinstrs -verify-misched -stop-before=machine-scheduler %s -o - | FileCheck %s --check-prefix=PRE
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32 -verify-machineinstrs -verify-misched -stop-after=machine-scheduler -debug-only=machine-scheduler %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DAG

; The real lowering path must produce one shared predicate and dead SCC defs,
; so the existing load-clustering mutation can form a cluster without removing
; any scheduler dependencies.
;
; PRE-LABEL: name: two_loads
; PRE: PRED_GLOBAL_LOAD_DWORD_RA_SADDR {{.*}}, 0, 0, [[PRED:%[0-9]+]], {{%[0-9]+}}, implicit-def dead $scc, implicit $exec
; PRE: PRED_GLOBAL_LOAD_DWORD_RA_SADDR {{.*}}, 4, 0, [[PRED]], {{%[0-9]+}}, implicit-def dead $scc, implicit $exec
;
; Identify the predicated loads by opcode, rather than accidentally checking
; the separate scalar kernarg-load cluster in this function.
; DAG: {{^}}SU([[FIRST:[0-9]+]]): {{.*}}PRED_GLOBAL_LOAD_DWORD_RA_SADDR
; DAG: Parent Cluster Index: [[CLUSTER:[0-9]+]]
; DAG: SU([[SECOND:[0-9]+]]): Ord  Latency=0 Cluster
; DAG: {{^}}SU([[SECOND]]): {{.*}}PRED_GLOBAL_LOAD_DWORD_RA_SADDR
; DAG: Parent Cluster Index: [[CLUSTER]]
; DAG: SU([[FIRST]]): Ord  Latency=0 Cluster

declare i32 @llvm.amdgcn.predicated.load.i32(ptr addrspace(1), i1, i32)

define amdgpu_kernel void @two_loads(ptr addrspace(1) %base, ptr addrspace(1) %out, i1 %pred) {
  %p1 = getelementptr i32, ptr addrspace(1) %base, i64 1
  %a = call i32 @llvm.amdgcn.predicated.load.i32(ptr addrspace(1) %base, i1 %pred, i32 0)
  %b = call i32 @llvm.amdgcn.predicated.load.i32(ptr addrspace(1) %p1, i1 %pred, i32 0)
  %sum = add i32 %a, %b
  store i32 %sum, ptr addrspace(1) %out
  ret void
}
