; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=SIVI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=SIVI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GETREG,GETREG-SDAG -check-prefix=GCN %s

declare i64 @llvm.readcyclecounter() #0

; GCN-LABEL: {{^}}test_readcyclecounter:
; MEMTIME-DAG: s_memtime s{{\[[0-9]+:[0-9]+\]}}
; GCN-DAG:     s_load_dwordx2
; GCN-DAG:     lgkmcnt
; MEMTIME:     store_dwordx2
; SIVI-NOT:    lgkmcnt
; MEMTIME:     s_memtime s{{\[[0-9]+:[0-9]+\]}}
; MEMTIME:     store_dwordx2

; GETREG-GISEL-DAG:  s_mov_b32 s[[SZERO:[0-9]+]], 0
; GETREG-GISEL-DAG:  v_mov_b32_e32 v[[ZERO:[0-9]+]], s[[SZERO]]
; GETREG-SDAG-DAG:  v_mov_b32_e32 v[[ZERO:[0-9]+]], 0
; GETREG-DAG:  s_getreg_b32 [[CNT1:s[0-9]+]], hwreg(HW_REG_SHADER_CYCLES, 0, 20)
; GETREG-DAG:  v_mov_b32_e32 v[[VCNT1:[0-9]+]], [[CNT1]]
; GETREG:      global_store_dwordx2 v{{.+}}, v[[[VCNT1]]:[[ZERO]]]
; GETREG:      s_getreg_b32 [[CNT2:s[0-9]+]], hwreg(HW_REG_SHADER_CYCLES, 0, 20)
; GETREG:      v_mov_b32_e32 v[[VCNT2:[0-9]+]], [[CNT2]]
; GETREG:      global_store_dwordx2 v{{.+}}, v[[[VCNT2]]:[[ZERO]]]

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
; GETREG-DAG:  s_getreg_b32 s1, hwreg(HW_REG_SHADER_CYCLES, 0, 20)
define amdgpu_cs i32 @test_readcyclecounter_smem(i64 addrspace(4)* inreg %in) #0 {
  %cycle0 = call i64 @llvm.readcyclecounter()
  %in.v = load i64, i64 addrspace(4)* %in
  %r.64 = add i64 %cycle0, %in.v
  %r.32 = trunc i64 %r.64 to i32
  ret i32 %r.32
}

attributes #0 = { nounwind }
