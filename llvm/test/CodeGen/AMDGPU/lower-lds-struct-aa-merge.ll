; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@b = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4

; CHECK-LABEL: @no_clobber_ds_load_stores_x2_preexisting_aa
; CHECK: store i32 1, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2_preexisting_aa.lds, align 16, !tbaa !1, !noalias !6
; CHECK: %val.a = load i32, ptr addrspace(3) %gep.a, align 4, !tbaa !1, !noalias !6
; CHECK: store i32 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2_preexisting_aa.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2_preexisting_aa.lds, i32 0, i32 1), align 16, !tbaa !1, !noalias !6
; CHECK: %val.b = load i32, ptr addrspace(3) %gep.b, align 4, !tbaa !1, !noalias !6

define amdgpu_kernel void @no_clobber_ds_load_stores_x2_preexisting_aa(ptr addrspace(1) %arg, i32 %i) {
bb:
  store i32 1, ptr addrspace(3) @a, align 4, !alias.scope !0, !noalias !3, !tbaa !5
  %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  %val.a = load i32, ptr addrspace(3) %gep.a, align 4, !alias.scope !0, !noalias !3, !tbaa !5
  store i32 2, ptr addrspace(3) @b, align 4, !alias.scope !3, !noalias !0, !tbaa !5
  %gep.b = getelementptr inbounds [64 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  %val.b = load i32, ptr addrspace(3) %gep.b, align 4, !alias.scope !3, !noalias !0, !tbaa !5
  %val = add i32 %val.a, %val.b
  store i32 %val, ptr addrspace(1) %arg, align 4
  ret void
}

!0 = !{!1}
!1 = distinct !{!1, !2}
!2 = distinct !{!2}
!3 = !{!4}
!4 = distinct !{!4, !2}
!5 = !{!6, !7, i64 0}
!6 = !{!"no_clobber_ds_load_stores_x2_preexisting_aa", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}

; CHECK:!0 = !{i64 0, i64 1}
; CHECK:!1 = !{!2, !3, i64 0}
; CHECK:!2 = !{!"no_clobber_ds_load_stores_x2_preexisting_aa", !3, i64 0}
; CHECK:!3 = !{!"int", !4, i64 0}
; CHECK:!4 = !{!"omnipotent char", !5, i64 0}
; CHECK:!5 = !{!"Simple C++ TBAA"}
; CHECK:!6 = !{}
