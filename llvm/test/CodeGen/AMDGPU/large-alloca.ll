; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}large_alloca:
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
; GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
; GCN: ScratchSize: 32776
define void @large_alloca(i32 addrspace(1)* %out, i32 %x, i32 %y) #0 {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 8191
  store i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 %y
  %load = load i32, i32* %gep1
  store i32 %load, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
