; RUN: opt -opaque-pointers=0 -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -opaque-pointers=0 -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@b = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4

; CHECK-LABEL: @no_clobber_ds_load_stores_x2_preexisting_aa
; CHECK: store i32 1, i32 addrspace(3)* %0, align 16, !tbaa !0, !noalias !5
; CHECK: %val.a = load i32, i32 addrspace(3)* %gep.a, align 4, !tbaa !0, !noalias !5
; CHECK: store i32 2, i32 addrspace(3)* %1, align 16, !tbaa !0, !noalias !5
; CHECK: %val.b = load i32, i32 addrspace(3)* %gep.b, align 4, !tbaa !0, !noalias !5

define amdgpu_kernel void @no_clobber_ds_load_stores_x2_preexisting_aa(i32 addrspace(1)* %arg, i32 %i) {
bb:
  store i32 1, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 0), align 4, !alias.scope !0, !noalias !3, !tbaa !5
  %gep.a = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 %i
  %val.a = load i32, i32 addrspace(3)* %gep.a, align 4, !alias.scope !0, !noalias !3, !tbaa !5
  store i32 2, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 0), align 4, !alias.scope !3, !noalias !0, !tbaa !5
  %gep.b = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 %i
  %val.b = load i32, i32 addrspace(3)* %gep.b, align 4, !alias.scope !3, !noalias !0, !tbaa !5
  %val = add i32 %val.a, %val.b
  store i32 %val, i32 addrspace(1)* %arg, align 4
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

; CHECK:!0 = !{!1, !2, i64 0}
; CHECK:!1 = !{!"no_clobber_ds_load_stores_x2_preexisting_aa", !2, i64 0}
; CHECK:!2 = !{!"int", !3, i64 0}
; CHECK:!3 = !{!"omnipotent char", !4, i64 0}
; CHECK:!4 = !{!"Simple C++ TBAA"}
; CHECK:!5 = !{}
