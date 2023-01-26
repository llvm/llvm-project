; RUN: llc -opaque-pointers=0 -march=amdgcn -mcpu=gfx900 -O3 < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -opaque-pointers=0 -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -opaque-pointers=0 -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@b = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@c = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4

; FIXME: Should combine the DS instructions into ds_write2 and ds_read2. This
; does not happen because when SILoadStoreOptimizer is run, the reads and writes
; are not adjacent. They are only moved later by MachineScheduler.

; GCN-LABEL: {{^}}no_clobber_ds_load_stores_x2:
; GCN: ds_write_b32
; GCN: ds_write_b32
; GCN: ds_read_b32
; GCN: ds_read_b32

; CHECK-LABEL: @no_clobber_ds_load_stores_x2
; CHECK: store i32 1, i32 addrspace(3)* %0, align 16, !alias.scope !0, !noalias !3
; CHECK: %val.a = load i32, i32 addrspace(3)* %gep.a, align 4, !alias.scope !0, !noalias !3
; CHECK: store i32 2, i32 addrspace(3)* %1, align 16, !alias.scope !3, !noalias !0
; CHECK: %val.b = load i32, i32 addrspace(3)* %gep.b, align 4, !alias.scope !3, !noalias !0

define amdgpu_kernel void @no_clobber_ds_load_stores_x2(i32 addrspace(1)* %arg, i32 %i) {
bb:
  store i32 1, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 0), align 4
  %gep.a = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 %i
  %val.a = load i32, i32 addrspace(3)* %gep.a, align 4
  store i32 2, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 0), align 4
  %gep.b = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 %i
  %val.b = load i32, i32 addrspace(3)* %gep.b, align 4
  %val = add i32 %val.a, %val.b
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}no_clobber_ds_load_stores_x3:
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_read_b32
; GCN-DAG: ds_read_b32
; GCN-DAG: ds_read_b32

; CHECK-LABEL: @no_clobber_ds_load_stores_x3
; CHECK: store i32 1, i32 addrspace(3)* %0, align 16, !alias.scope !5, !noalias !8
; CHECK: %val.a = load i32, i32 addrspace(3)* %gep.a, align 4, !alias.scope !5, !noalias !8
; CHECK: store i32 2, i32 addrspace(3)* %1, align 16, !alias.scope !11, !noalias !12
; CHECK: %val.b = load i32, i32 addrspace(3)* %gep.b, align 4, !alias.scope !11, !noalias !12
; CHECK: store i32 3, i32 addrspace(3)* %2, align 16, !alias.scope !13, !noalias !14
; CHECK: %val.c = load i32, i32 addrspace(3)* %gep.c, align 4, !alias.scope !13, !noalias !14

define amdgpu_kernel void @no_clobber_ds_load_stores_x3(i32 addrspace(1)* %arg, i32 %i) {
bb:
  store i32 1, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 0), align 4
  %gep.a = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @a, i32 0, i32 %i
  %val.a = load i32, i32 addrspace(3)* %gep.a, align 4
  store i32 2, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 0), align 4
  %gep.b = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @b, i32 0, i32 %i
  %val.b = load i32, i32 addrspace(3)* %gep.b, align 4
  store i32 3, i32 addrspace(3)* getelementptr inbounds ([64 x i32], [64 x i32] addrspace(3)* @c, i32 0, i32 0), align 4
  %gep.c = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* @c, i32 0, i32 %i
  %val.c = load i32, i32 addrspace(3)* %gep.c, align 4
  %val.1 = add i32 %val.a, %val.b
  %val = add i32 %val.1, %val.c
  store i32 %val, i32 addrspace(1)* %arg, align 4
  ret void
}

; CHECK: !0 = !{!1}
; CHECK: !1 = distinct !{!1, !2}
; CHECK: !2 = distinct !{!2}
; CHECK: !3 = !{!4}
; CHECK: !4 = distinct !{!4, !2}
; CHECK: !5 = !{!6}
; CHECK: !6 = distinct !{!6, !7}
; CHECK: !7 = distinct !{!7}
; CHECK: !8 = !{!9, !10}
; CHECK: !9 = distinct !{!9, !7}
; CHECK: !10 = distinct !{!10, !7}
; CHECK: !11 = !{!9}
; CHECK: !12 = !{!6, !10}
; CHECK: !13 = !{!10}
; CHECK: !14 = !{!6, !9}
