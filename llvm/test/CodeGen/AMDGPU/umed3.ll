; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SICIVI -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SICIVI -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x() #0

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_i32:
; GCN: v_med3_u32 v{{[0-9]+}}, v{{[0-9]+}}, 12, 17
define amdgpu_kernel void @v_test_umed3_r_i_i_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp ugt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp ult i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_multi_use_r_i_i_i32:
; GCN: v_max_u32
; GCN: v_min_u32
define amdgpu_kernel void @v_test_umed3_multi_use_r_i_i_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp ugt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp ult i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store volatile i32 %i0, i32 addrspace(1)* %outgep
  store volatile i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_constant_order_i32:
; GCN: v_max_u32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
; GCN: v_min_u32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
define amdgpu_kernel void @v_test_umed3_r_i_i_constant_order_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp ugt i32 %a, 17
  %i0 = select i1 %icmp0, i32 %a, i32 17

  %icmp1 = icmp ult i32 %i0, 12
  %i1 = select i1 %icmp1, i32 %i0, i32 12

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_sign_mismatch_i32:
; GCN: v_max_i32_e32 v{{[0-9]+}}, 12, v{{[0-9]+}}
; GCN: v_min_u32_e32 v{{[0-9]+}}, 17, v{{[0-9]+}}
define amdgpu_kernel void @v_test_umed3_r_i_i_sign_mismatch_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0

  %icmp0 = icmp sgt i32 %a, 12
  %i0 = select i1 %icmp0, i32 %a, i32 12

  %icmp1 = icmp ult i32 %i0, 17
  %i1 = select i1 %icmp1, i32 %i0, i32 17

  store i32 %i1, i32 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_i64:
; GCN: v_cmp_lt_u64
; GCN: v_cmp_gt_u64
define amdgpu_kernel void @v_test_umed3_r_i_i_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep0

  %icmp0 = icmp ugt i64 %a, 12
  %i0 = select i1 %icmp0, i64 %a, i64 12

  %icmp1 = icmp ult i64 %i0, 17
  %i1 = select i1 %icmp1, i64 %i0, i64 17

  store i64 %i1, i64 addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_r_i_i_i16:
; SICIVI: v_med3_u32 v{{[0-9]+}}, v{{[0-9]+}}, 12, 17
; GFX9: v_med3_u16 v{{[0-9]+}}, v{{[0-9]+}}, 12, 17
define amdgpu_kernel void @v_test_umed3_r_i_i_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep0

  %icmp0 = icmp ugt i16 %a, 12
  %i0 = select i1 %icmp0, i16 %a, i16 12

  %icmp1 = icmp ult i16 %i0, 17
  %i1 = select i1 %icmp1, i16 %i0, i16 17

  store i16 %i1, i16 addrspace(1)* %outgep
  ret void
}

define internal i32 @umin(i32 %x, i32 %y) #2 {
  %cmp = icmp ult i32 %x, %y
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel
}

define internal i32 @umax(i32 %x, i32 %y) #2 {
  %cmp = icmp ugt i32 %x, %y
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel
}

define internal i16 @umin16(i16 %x, i16 %y) #2 {
  %cmp = icmp ult i16 %x, %y
  %sel = select i1 %cmp, i16 %x, i16 %y
  ret i16 %sel
}

define internal i16 @umax16(i16 %x, i16 %y) #2 {
  %cmp = icmp ugt i16 %x, %y
  %sel = select i1 %cmp, i16 %x, i16 %y
  ret i16 %sel
}

define internal i8 @umin8(i8 %x, i8 %y) #2 {
  %cmp = icmp ult i8 %x, %y
  %sel = select i1 %cmp, i8 %x, i8 %y
  ret i8 %sel
}

define internal i8 @umax8(i8 %x, i8 %y) #2 {
  %cmp = icmp ugt i8 %x, %y
  %sel = select i1 %cmp, i8 %x, i8 %y
  ret i8 %sel
}

; 16 combinations

; 0: max(min(x, y), min(max(x, y), z))
; 1: max(min(x, y), min(max(y, x), z))
; 2: max(min(x, y), min(z, max(x, y)))
; 3: max(min(x, y), min(z, max(y, x)))
; 4: max(min(y, x), min(max(x, y), z))
; 5: max(min(y, x), min(max(y, x), z))
; 6: max(min(y, x), min(z, max(x, y)))
; 7: max(min(y, x), min(z, max(y, x)))
;
; + commute outermost max


; FIXME: In these cases we probably should have used scalar operations
; instead.

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_0(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_1:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_1(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_2:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_2(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_3:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_3(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_4:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_4(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_5:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_5(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_6:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_6(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_7:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_7(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_8:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_8(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_9:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_9(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_10:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_10(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_11:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_11(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_12:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_12(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_13:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_13(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_14:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_14(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_15:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_15(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %y, i32 %x)
  %tmp1 = call i32 @umax(i32 %y, i32 %x)
  %tmp2 = call i32 @umin(i32 %z, i32 %tmp1)
  %tmp3 = call i32 @umax(i32 %tmp2, i32 %tmp0)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i16_pat_0:
; GCN: s_and_b32
; GCN: s_and_b32
; GCN: s_and_b32
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i16_pat_0(i16 addrspace(1)* %arg, i16 %x, i16 %y, i16 %z) #1 {
bb:
  %tmp0 = call i16 @umin16(i16 %x, i16 %y)
  %tmp1 = call i16 @umax16(i16 %x, i16 %y)
  %tmp2 = call i16 @umin16(i16 %tmp1, i16 %z)
  %tmp3 = call i16 @umax16(i16 %tmp0, i16 %tmp2)
  store i16 %tmp3, i16 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i8_pat_0:
; GCN: s_and_b32
; GCN: s_and_b32
; GCN: s_and_b32
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i8_pat_0(i8 addrspace(1)* %arg, i8 %x, i8 %y, i8 %z) #1 {
bb:
  %tmp0 = call i8 @umin8(i8 %x, i8 %y)
  %tmp1 = call i8 @umax8(i8 %x, i8 %y)
  %tmp2 = call i8 @umin8(i8 %tmp1, i8 %z)
  %tmp3 = call i8 @umax8(i8 %tmp0, i8 %tmp2)
  store i8 %tmp3, i8 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_multi_use_0:
; GCN-NOT: v_med3_u32
define amdgpu_kernel void @s_test_umed3_i32_pat_0_multi_use_0(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store volatile i32 %tmp0, i32 addrspace(1)* %arg
  store volatile i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_multi_use_1:
; GCN-NOT: v_med3_u32
define amdgpu_kernel void @s_test_umed3_i32_pat_0_multi_use_1(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store volatile i32 %tmp1, i32 addrspace(1)* %arg
  store volatile i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_multi_use_2:
; GCN-NOT: v_med3_u32
define amdgpu_kernel void @s_test_umed3_i32_pat_0_multi_use_2(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store volatile i32 %tmp2, i32 addrspace(1)* %arg
  store volatile i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_multi_use_result:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_0_multi_use_result(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store volatile i32 %tmp3, i32 addrspace(1)* %arg
  store volatile i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_imm_src0:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, 1, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_0_imm_src0(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 1, i32 %y)
  %tmp1 = call i32 @umax(i32 1, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_imm_src1:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, 2, v{{[0-9]+}}
define amdgpu_kernel void @s_test_umed3_i32_pat_0_imm_src1(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 2)
  %tmp1 = call i32 @umax(i32 %x, i32 2)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 %z)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}s_test_umed3_i32_pat_0_imm_src2:
; GCN: v_med3_u32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, 9
define amdgpu_kernel void @s_test_umed3_i32_pat_0_imm_src2(i32 addrspace(1)* %arg, i32 %x, i32 %y, i32 %z) #1 {
bb:
  %tmp0 = call i32 @umin(i32 %x, i32 %y)
  %tmp1 = call i32 @umax(i32 %x, i32 %y)
  %tmp2 = call i32 @umin(i32 %tmp1, i32 9)
  %tmp3 = call i32 @umax(i32 %tmp0, i32 %tmp2)
  store i32 %tmp3, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_test_umed3_i16_pat_0:
; SI: v_med3_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; FIXME: VI not matching med3
; VI: v_min_u16
; VI: v_max_u16
; VI: v_min_u16
; VI: v_max_u16

; GFX9: v_med3_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_test_umed3_i16_pat_0(i16 addrspace(1)* %arg, i16 addrspace(1)* %out, i16 addrspace(1)* %a.ptr) #1 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr inbounds i16, i16 addrspace(1)* %a.ptr, i32 %tid
  %gep1 = getelementptr inbounds i16, i16 addrspace(1)* %gep0, i32 3
  %gep2 = getelementptr inbounds i16, i16 addrspace(1)* %gep0, i32 8
  %out.gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %x = load i16, i16 addrspace(1)* %gep0
  %y = load i16, i16 addrspace(1)* %gep1
  %z = load i16, i16 addrspace(1)* %gep2

  %tmp0 = call i16 @umin16(i16 %x, i16 %y)
  %tmp1 = call i16 @umax16(i16 %x, i16 %y)
  %tmp2 = call i16 @umin16(i16 %tmp1, i16 %z)
  %tmp3 = call i16 @umax16(i16 %tmp0, i16 %tmp2)
  store i16 %tmp3, i16 addrspace(1)* %out.gep
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone alwaysinline }
