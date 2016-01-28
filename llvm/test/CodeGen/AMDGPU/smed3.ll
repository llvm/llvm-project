; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.r600.read.tidig.x() #0

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_i32:
; GCN: v_med3_i32 v{{[0-9]+}}, v{{[0-9]+}}, 12, 17
define void @v_test_smed3_r_i_i_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp sgt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp slt i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_multi_use_r_i_i_i32:
; GCN: v_max_i32
; GCN: v_min_i32
define void @v_test_smed3_multi_use_r_i_i_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp sgt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp slt i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store volatile i32 %i0, i32 addrspace(1)* %outgep
  store volatile i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_constant_order_i32:
; GCN: v_max_i32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
; GCN: v_min_i32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
define void @v_test_smed3_r_i_i_constant_order_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp sgt i32 %a, 17
  %i0 = select i1 %icmp0, i32 %a, i32 17

  %icmp1 = icmp slt i32 %i0, 12
  %i1 = select i1 %icmp1, i32 %i0, i32 12

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_sign_mismatch_i32:
; GCN: v_max_u32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
; GCN: v_min_i32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
define void @v_test_smed3_r_i_i_sign_mismatch_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp ugt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp slt i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_i64:
; GCN: v_cmp_lt_i64
; GCN: v_cmp_gt_i64
define void @v_test_smed3_r_i_i_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep0

  %icmp0 = icmp sgt i64 %a, 12
  %i0 = select i1 %icmp0, i64 %a, i64 12

  %icmp1 = icmp slt i64 %i0, 17
  %i1 = select i1 %icmp1, i64 %i0, i64 17

  store i64 %i1, i64 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_smed3_r_i_i_i16:
; GCN: v_med3_i32 v{{[0-9]+}}, v{{[0-9]+}}, 12, 17
define void @v_test_smed3_r_i_i_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0

  %icmp0 = icmp sgt i16 %a, 12
  %i0 = select i1 %icmp0, i16 %a, i16 12

  %icmp1 = icmp slt i16 %i0, 17
  %i1 = select i1 %icmp1, i16 %i0, i16 17

  store i16 %i1, i16 addrspace(1)* %outgep
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
