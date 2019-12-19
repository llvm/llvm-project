; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck --enable-var-scope --check-prefixes=CHECK,DBG %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=CHECK,GCN %s
; REQUIRES: asserts

; CHECK-LABEL: {{^}}cluster_load_cluster_store:
define amdgpu_kernel void @cluster_load_cluster_store(i32* noalias %lb, i32* noalias %sb) {
bb:
; DBG: Cluster ld/st SU(1) - SU(2)

; DBG: Cluster ld/st SU([[L1:[0-9]+]]) - SU([[L2:[0-9]+]])
; DBG: Cluster ld/st SU([[L2]]) - SU([[L3:[0-9]+]])
; DBG: Cluster ld/st SU([[L3]]) - SU([[L4:[0-9]+]])
; GCN:      flat_load_dword [[LD1:v[0-9]+]], v[{{[0-9:]+}}]
; GCN-NEXT: flat_load_dword [[LD2:v[0-9]+]], v[{{[0-9:]+}}] offset:8
; GCN-NEXT: flat_load_dword [[LD3:v[0-9]+]], v[{{[0-9:]+}}] offset:16
; GCN-NEXT: flat_load_dword [[LD4:v[0-9]+]], v[{{[0-9:]+}}] offset:24
  %la0 = getelementptr inbounds i32, i32* %lb, i32 0
  %ld0 = load i32, i32* %la0
  %la1 = getelementptr inbounds i32, i32* %lb, i32 2
  %ld1 = load i32, i32* %la1
  %la2 = getelementptr inbounds i32, i32* %lb, i32 4
  %ld2 = load i32, i32* %la2
  %la3 = getelementptr inbounds i32, i32* %lb, i32 6
  %ld3 = load i32, i32* %la3

; DBG: Cluster ld/st SU([[S1:[0-9]+]]) - SU([[S2:[0-9]+]])
; DBG: Cluster ld/st SU([[S2]]) - SU([[S3:[0-9]+]])
; DBG: Cluster ld/st SU([[S3]]) - SU([[S4:[0-9]+]])
; GCN:      flat_store_dword v[{{[0-9:]+}}], [[LD1]]
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD2]] offset:8
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD3]] offset:16
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD4]] offset:24
  %sa0 = getelementptr inbounds i32, i32* %sb, i32 0
  store i32 %ld0, i32* %sa0
  %sa1 = getelementptr inbounds i32, i32* %sb, i32 2
  store i32 %ld1, i32* %sa1
  %sa2 = getelementptr inbounds i32, i32* %sb, i32 4
  store i32 %ld2, i32* %sa2
  %sa3 = getelementptr inbounds i32, i32* %sb, i32 6
  store i32 %ld3, i32* %sa3

  ret void
}

; CHECK-LABEL: {{^}}cluster_load_valu_cluster_store:
define amdgpu_kernel void @cluster_load_valu_cluster_store(i32* noalias %lb, i32* noalias %sb) {
bb:
; DBG: Cluster ld/st SU(1) - SU(2)

; DBG: Cluster ld/st SU([[L1:[0-9]+]]) - SU([[L2:[0-9]+]])
; DBG: Cluster ld/st SU([[L2]]) - SU([[L3:[0-9]+]])
; DBG: Cluster ld/st SU([[L3]]) - SU([[L4:[0-9]+]])
; GCN:      flat_load_dword [[LD1:v[0-9]+]], v[{{[0-9:]+}}]
; GCN-NEXT: flat_load_dword [[LD2:v[0-9]+]], v[{{[0-9:]+}}] offset:8
; GCN-NEXT: flat_load_dword [[LD3:v[0-9]+]], v[{{[0-9:]+}}] offset:16
; GCN-NEXT: flat_load_dword [[LD4:v[0-9]+]], v[{{[0-9:]+}}] offset:24
  %la0 = getelementptr inbounds i32, i32* %lb, i32 0
  %ld0 = load i32, i32* %la0
  %la1 = getelementptr inbounds i32, i32* %lb, i32 2
  %ld1 = load i32, i32* %la1
  %la2 = getelementptr inbounds i32, i32* %lb, i32 4
  %ld2 = load i32, i32* %la2
  %la3 = getelementptr inbounds i32, i32* %lb, i32 6
  %ld3 = load i32, i32* %la3

; DBG: Cluster ld/st SU([[S1:[0-9]+]]) - SU([[S2:[0-9]+]])
; DBG: Cluster ld/st SU([[S2]]) - SU([[S3:[0-9]+]])
; DBG: Cluster ld/st SU([[S3]]) - SU([[S4:[0-9]+]])
; GCN:      v_add_u32_e32 [[ST2:v[0-9]+]], 1, [[LD2]]
; GCN:      flat_store_dword v[{{[0-9:]+}}], [[LD1]]
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[ST2]] offset:8
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD3]] offset:16
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD4]] offset:24
  %sa0 = getelementptr inbounds i32, i32* %sb, i32 0
  store i32 %ld0, i32* %sa0
  %sa1 = getelementptr inbounds i32, i32* %sb, i32 2
  %add = add i32 %ld1, 1
  store i32 %add, i32* %sa1
  %sa2 = getelementptr inbounds i32, i32* %sb, i32 4
  store i32 %ld2, i32* %sa2
  %sa3 = getelementptr inbounds i32, i32* %sb, i32 6
  store i32 %ld3, i32* %sa3

  ret void
}
