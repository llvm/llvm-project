; RUN: llc -march=amdgcn -mcpu=gfx900 -O3 --amdgpu-lower-module-lds-strategy=module < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

%vec_type = type { %vec_base }
%vec_base = type { %union.anon }
%union.anon = type { %"vec_base<char, 3>::n_vec_" }
%"vec_base<char, 3>::n_vec_" = type { [3 x i8] }

$_f1 = comdat any
$_f2 = comdat any
@_f1 = linkonce_odr hidden local_unnamed_addr addrspace(3) global %vec_type undef, comdat, align 1
@_f2 = linkonce_odr hidden local_unnamed_addr addrspace(3) global %vec_type undef, comdat, align 1

;.
; CHECK: @[[LLVM_AMDGCN_KERNEL_TEST_LDS:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(3) global [[LLVM_AMDGCN_KERNEL_TEST_LDS_T:%.*]] undef, align 4, !absolute_symbol !0
;.
define protected amdgpu_kernel void @test(ptr addrspace(1) nocapture %ptr.coerce) local_unnamed_addr #0 {
; GCN-LABEL: test:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    v_mov_b32_e32 v1, 2
; GCN-NEXT:    ds_write_b8 v0, v1
; GCN-NEXT:    ds_read_u8 v2, v0 offset:2
; GCN-NEXT:    ds_read_u16 v3, v0
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    ds_write_b8 v0, v2 offset:6
; GCN-NEXT:    ds_write_b16 v0, v3 offset:4
; GCN-NEXT:    v_cmp_eq_u16_sdwa s[2:3], v3, v1 src0_sel:BYTE_0 src1_sel:DWORD
; GCN-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[2:3]
; GCN-NEXT:    global_store_byte v0, v1, s[0:1]
; GCN-NEXT:    s_endpgm
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i8 3, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, align 4, !alias.scope !1, !noalias !4
; CHECK-NEXT:    tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noundef align 1 dereferenceable(3) getelementptr inbounds (%llvm.amdgcn.kernel.test.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, i32 0, i32 2), ptr addrspace(3) noundef align 1 dereferenceable(3) @llvm.amdgcn.kernel.test.lds, i64 3, i1 false), !alias.scope !6, !noalias !7
; CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.test.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, i32 0, i32 2), align 4, !alias.scope !4, !noalias !1
; CHECK-NEXT:    [[CMP_I_I:%.*]] = icmp eq i8 [[TMP4]], 3
; CHECK-NEXT:    store i8 2, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, align 4, !alias.scope !1, !noalias !4
; CHECK-NEXT: tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noundef align 1 dereferenceable(3) getelementptr inbounds (%llvm.amdgcn.kernel.test.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, i32 0, i32 2), ptr addrspace(3) noundef align 1 dereferenceable(3) @llvm.amdgcn.kernel.test.lds, i64 3, i1 false), !alias.scope !6, !noalias !7
; CHECK-NEXT:    [[TMP9:%.*]] = load i8, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.test.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.test.lds, i32 0, i32 2), align 4, !alias.scope !4, !noalias !1
; CHECK-NEXT:    [[CMP_I_I19:%.*]] = icmp eq i8 [[TMP9]], 2
; CHECK-NEXT:    [[TMP10:%.*]] = and i1 [[CMP_I_I19]], [[CMP_I_I]]
; CHECK-NEXT:    [[FROMBOOL8:%.*]] = zext i1 [[TMP10]] to i8
; CHECK-NEXT:    store i8 [[FROMBOOL8]], ptr addrspace(1) [[PTR_COERCE:%.*]], align 1
; CHECK-NEXT:    ret void
;
entry:
  store i8 3, ptr addrspace(3) @_f1, align 1
  tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noundef align 1 dereferenceable(3) @_f2, ptr addrspace(3) noundef align 1 dereferenceable(3) @_f1, i64 3, i1 false)
  %0 = load i8, ptr addrspace(3) @_f2, align 1
  %cmp.i.i = icmp eq i8 %0, 3
  store i8 2, ptr addrspace(3) @_f1, align 1
  tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noundef align 1 dereferenceable(3) @_f2, ptr addrspace(3) noundef align 1 dereferenceable(3) @_f1, i64 3, i1 false)
  %1 = load i8, ptr addrspace(3) @_f2, align 1
  %cmp.i.i19 = icmp eq i8 %1, 2
  %2 = and i1 %cmp.i.i19, %cmp.i.i
  %frombool8 = zext i1 %2 to i8
  store i8 %frombool8, ptr addrspace(1) %ptr.coerce, align 1
  ret void
}

declare void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i64, i1 immarg) #1

;.
; CHECK: attributes #[[ATTR0:[0-9]+]] = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
;.
; CHECK: [[META0:![0-9]+]] = !{i64 0, i64 1}
; CHECK: [[META1:![0-9]+]] = !{!2}
; CHECK: [[META2:![0-9]+]] = distinct !{!2, !3}
; CHECK: [[META3:![0-9]+]] = distinct !{!3}
; CHECK: [[META4:![0-9]+]] = !{!5}
; CHECK: [[META5:![0-9]+]] = distinct !{!5, !3}
; CHECK: [[META6:![0-9]+]] = !{!5, !2}
; CHECK: [[META7:![0-9]+]] = !{}
;.
