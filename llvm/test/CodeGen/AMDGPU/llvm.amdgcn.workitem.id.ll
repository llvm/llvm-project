; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=CO-V2 -check-prefix=CI-HSA  %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=CO-V2 -check-prefix=VI-HSA  %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=MESA -check-prefix=SI-MESA %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=MESA -check-prefix=VI-MESA %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2,SI-MESA %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2,VI-MESA %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 132{{$}}

; ALL-LABEL {{^}}test_workitem_id_x:
; CO-V2: enable_vgpr_workitem_id = 0

; ALL-NOT: v0
; ALL: {{buffer|flat}}_store_dword {{.*}}v0
define void @test_workitem_id_x(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 2180{{$}}

; ALL-LABEL {{^}}test_workitem_id_y:
; CO-V2: enable_vgpr_workitem_id = 1

; ALL-NOT: v1
; ALL: {{buffer|flat}}_store_dword {{.*}}v1
define void @test_workitem_id_y(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 4228{{$}}

; ALL-LABEL {{^}}test_workitem_id_z:
; CO-V2: enable_vgpr_workitem_id = 2

; ALL-NOT: v2
; ALL: {{buffer|flat}}_store_dword {{.*}}v2
define void @test_workitem_id_z(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
