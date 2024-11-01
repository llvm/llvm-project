; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-unknown < %s | FileCheck %s

; and can be eliminated
; CHECK-LABEL: {{^}}test_workitem_id_x_known_max_range:
; CHECK-NOT: v0
; CHECK: {{flat|buffer}}_store_dword {{.*}}v0
define amdgpu_kernel void @test_workitem_id_x_known_max_range(ptr addrspace(1) nocapture %out) #0 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0
  %and = and i32 %id, 1023
  store i32 %and, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_workitem_id_x_known_trunc_1_bit_range:
; CHECK-NOT: v_and_b32
; CHECK: {{flat|buffer}}_store_dword {{.*}}v0
define amdgpu_kernel void @test_workitem_id_x_known_trunc_1_bit_range(ptr addrspace(1) nocapture %out) #0 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0
  %and = and i32 %id, 511
  store i32 %and, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}test_workitem_id_x_known_max_range_m1:
; CHECK-NOT: v0
; CHECK-NOT: v_and_b32
; CHECK: {{flat|buffer}}_store_dword {{.*}}v0
define amdgpu_kernel void @test_workitem_id_x_known_max_range_m1(ptr addrspace(1) nocapture %out) #0 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !1
  %and = and i32 %id, 255
  store i32 %and, ptr addrspace(1) %out, align 4
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { norecurse nounwind "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind readnone }

!0 = !{i32 0, i32 1024}
!1 = !{i32 0, i32 1023}
