; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mattr=-promote-alloca,+max-private-element-size-16 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=HSA-ELT16,ALL %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mattr=-promote-alloca,+max-private-element-size-8 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=HSA-ELT8,ALL,HSA-ELTGE8 %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mattr=-promote-alloca,+max-private-element-size-4 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap --check-prefixes=HSA-ELT4,ALL %s


; ALL-LABEL: {{^}}private_elt_size_v4i32:

; HSA-ELT16: private_element_size = 3
; HSA-ELT8: private_element_size = 2
; HSA-ELT4: private_element_size = 1


; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:16
; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32
; HSA-ELT16-DAG: buffer_load_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}

; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:24{{$}}
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:16
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:40

; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen
; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen


; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:16{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:20{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:24{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:28{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:32{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:36{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:40{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:44{{$}}

; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:4{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:8{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:12{{$}}
define amdgpu_kernel void @private_elt_size_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %index.array) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %tid to i64
  %gep.index = getelementptr inbounds i32, ptr addrspace(1) %index.array, i64 %idxprom
  %index.load = load i32, ptr addrspace(1) %gep.index
  %index = and i32 %index.load, 2
  %alloca = alloca [2 x <4 x i32>], align 16, addrspace(5)
  %gep1 = getelementptr inbounds [2 x <4 x i32>], ptr addrspace(5) %alloca, i32 0, i32 1
  store <4 x i32> zeroinitializer, ptr addrspace(5) %alloca
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x <4 x i32>], ptr addrspace(5) %alloca, i32 0, i32 %index
  %load = load <4 x i32>, ptr addrspace(5) %gep2
  store <4 x i32> %load, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}private_elt_size_v8i32:
; HSA-ELT16: private_element_size = 3
; HSA-ELT8: private_element_size = 2
; HSA-ELT4: private_element_size = 1

; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32
; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:48
; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:64
; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:80

; HSA-ELT16-DAG: buffer_load_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}
; HSA-ELT16-DAG: buffer_load_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}


; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:40
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:48
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:56
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:88
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:80
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:72
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:64

; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen
; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen


; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:32{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:36{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:40{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:44{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:48{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:52{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:56{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:60{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:64{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:68{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:72{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:76{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:80{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:84{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:88{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:92{{$}}

; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:4{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:8{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:12{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:16{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:20{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:24{{$}}
; HSA-ELT4-DAG: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:28{{$}}
define amdgpu_kernel void @private_elt_size_v8i32(ptr addrspace(1) %out, ptr addrspace(1) %index.array) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %tid to i64
  %gep.index = getelementptr inbounds i32, ptr addrspace(1) %index.array, i64 %idxprom
  %index.load = load i32, ptr addrspace(1) %gep.index
  %index = and i32 %index.load, 2
  %alloca = alloca [2 x <8 x i32>], align 32, addrspace(5)
  %gep1 = getelementptr inbounds [2 x <8 x i32>], ptr addrspace(5) %alloca, i32 0, i32 1
  store <8 x i32> zeroinitializer, ptr addrspace(5) %alloca
  store <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x <8 x i32>], ptr addrspace(5) %alloca, i32 0, i32 %index
  %load = load <8 x i32>, ptr addrspace(5) %gep2
  store <8 x i32> %load, ptr addrspace(1) %out
  ret void
}


; ALL-LABEL: {{^}}private_elt_size_i64:
; HSA-ELT16: private_element_size = 3
; HSA-ELT8: private_element_size = 2
; HSA-ELT4: private_element_size = 1

; HSA-ELTGE8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{off|v[0-9]}}, s[0:3], 0 offset:1
; HSA-ELTGE8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{off|v[0-9]}}, s[0:3], 0 offset:2
; HSA-ELTGE8-DAG: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, s[0:3], 0 offen
; HSA-ELTGE8: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]


; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:16{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:20{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:24{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:28{{$}}

; HSA-ELT4-DAG: buffer_load_dword v[[HI:[0-9]+]], v{{[0-9]+}}, s[0:3], 0 offen offset:4{{$}}
; HSA-ELT4-DAG: buffer_load_dword v[[LO:[0-9]+]], v{{[0-9]+}}, s[0:3], 0 offen{{$}}
; HSA-ELT4: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @private_elt_size_i64(ptr addrspace(1) %out, ptr addrspace(1) %index.array) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %tid to i64
  %gep.index = getelementptr inbounds i32, ptr addrspace(1) %index.array, i64 %idxprom
  %index.load = load i32, ptr addrspace(1) %gep.index
  %index = and i32 %index.load, 2
  %alloca = alloca [2 x i64], align 16, addrspace(5)
  %gep1 = getelementptr inbounds [2 x i64], ptr addrspace(5) %alloca, i32 0, i32 1
  store i64 0, ptr addrspace(5) %alloca
  store i64 34359738602, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x i64], ptr addrspace(5) %alloca, i32 0, i32 %index
  %load = load i64, ptr addrspace(5) %gep2
  store i64 %load, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}private_elt_size_f64:
; HSA-ELT16: private_element_size = 3
; HSA-ELT8: private_element_size = 2
; HSA-ELT4: private_element_size = 1

; HSA-ELTGE8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:16
; HSA-ELTGE8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:24
; HSA-ELTGE8-DAG: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, s[0:3], 0 offen
; HSA-ELTGE8: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]


; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:16{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:20{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:24{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:28{{$}}

; HSA-ELT4-DAG: buffer_load_dword v[[HI:[0-9]+]], v{{[0-9]+}}, s[0:3], 0 offen offset:4{{$}}
; HSA-ELT4-DAG: buffer_load_dword v[[LO:[0-9]+]], v{{[0-9]+}}, s[0:3], 0 offen{{$}}
; HSA-ELT4: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @private_elt_size_f64(ptr addrspace(1) %out, ptr addrspace(1) %index.array) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %tid to i64
  %gep.index = getelementptr inbounds i32, ptr addrspace(1) %index.array, i64 %idxprom
  %index.load = load i32, ptr addrspace(1) %gep.index
  %index = and i32 %index.load, 2
  %alloca = alloca [2 x double], align 16, addrspace(5)
  %gep1 = getelementptr inbounds [2 x double], ptr addrspace(5) %alloca, i32 0, i32 1
  store double 0.0, ptr addrspace(5) %alloca
  store double 4.0, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x double], ptr addrspace(5) %alloca, i32 0, i32 %index
  %load = load double, ptr addrspace(5) %gep2
  store double %load, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}private_elt_size_v2i64:
; HSA-ELT16: private_element_size = 3
; HSA-ELT8: private_element_size = 2
; HSA-ELT4: private_element_size = 1

; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:16
; HSA-ELT16-DAG: buffer_store_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32
; HSA-ELT16-DAG: buffer_load_dwordx4 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}

; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:16{{$}}
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:24
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:40
; HSA-ELT8-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 0 offset:32

; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen
; HSA-ELT8: buffer_load_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], 0 offen


; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:16{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:20{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:24{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:28{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:32{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:36{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:40{{$}}
; HSA-ELT4-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], 0 offset:44{{$}}

; HSA-ELT4: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:12{{$}}
; HSA-ELT4: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:8{{$}}
; HSA-ELT4: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen offset:4{{$}}
; HSA-ELT4: buffer_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen{{$}}
define amdgpu_kernel void @private_elt_size_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %index.array) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %tid to i64
  %gep.index = getelementptr inbounds i32, ptr addrspace(1) %index.array, i64 %idxprom
  %index.load = load i32, ptr addrspace(1) %gep.index
  %index = and i32 %index.load, 2
  %alloca = alloca [2 x <2 x i64>], align 16, addrspace(5)
  %gep1 = getelementptr inbounds [2 x <2 x i64>], ptr addrspace(5) %alloca, i32 0, i32 1
  store <2 x i64> zeroinitializer, ptr addrspace(5) %alloca
  store <2 x i64> <i64 1, i64 2>, ptr addrspace(5) %gep1
  %gep2 = getelementptr inbounds [2 x <2 x i64>], ptr addrspace(5) %alloca, i32 0, i32 %index
  %load = load <2 x i64>, ptr addrspace(5) %gep2
  store <2 x i64> %load, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
