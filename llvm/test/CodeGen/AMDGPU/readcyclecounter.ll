; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=SIVI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=SIVI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=GCN %s

declare i64 @llvm.readcyclecounter() #0

; GCN-LABEL: {{^}}test_readcyclecounter:
; MEMTIME-DAG: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN-DAG:     s_load_dwordx2
; GCN-DAG:     lgkmcnt
; MEMTIME:     store_dwordx2
; SIVI-NOT:    lgkmcnt
; MEMTIME:     s_memtime s{{\[[0-9]+:[0-9]+\]}}
; MEMTIME:     store_dwordx2

define amdgpu_kernel void @test_readcyclecounter(i64 addrspace(1)* %out) #0 {
  %cycle0 = call i64 @llvm.readcyclecounter()
  store volatile i64 %cycle0, i64 addrspace(1)* %out

  %cycle1 = call i64 @llvm.readcyclecounter()
  store volatile i64 %cycle1, i64 addrspace(1)* %out
  ret void
}

; This test used to crash in ScheduleDAG.
;
; GCN-LABEL: {{^}}test_readcyclecounter_smem:
; MEMTIME-DAG: s_memtime
; GCN-DAG:     s_load_dword
define amdgpu_cs i32 @test_readcyclecounter_smem(i64 addrspace(4)* inreg %in) #0 {
  %cycle0 = call i64 @llvm.readcyclecounter()
  %in.v = load i64, i64 addrspace(4)* %in
  %r.64 = add i64 %cycle0, %in.v
  %r.32 = trunc i64 %r.64 to i32
  ret i32 %r.32
}

attributes #0 = { nounwind }
