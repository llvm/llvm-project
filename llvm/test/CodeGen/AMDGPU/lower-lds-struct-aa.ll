; RUN: llc -march=amdgcn -mcpu=gfx900 -O3 < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

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
; CHECK: store i32 1, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, align 16, !alias.scope !1, !noalias !4
; CHECK: %val.a = load i32, ptr addrspace(3) %gep.a, align 4, !alias.scope !1, !noalias !4
; CHECK: store i32 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, i32 0, i32 1), align 16, !alias.scope !4, !noalias !1
; CHECK: %val.b = load i32, ptr addrspace(3) %gep.b, align 4, !alias.scope !4, !noalias !1

define amdgpu_kernel void @no_clobber_ds_load_stores_x2(ptr addrspace(1) %arg, i32 %i) {
bb:
  store i32 1, ptr addrspace(3) @a, align 4
  %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  %val.a = load i32, ptr addrspace(3) %gep.a, align 4
  store i32 2, ptr addrspace(3) @b, align 4
  %gep.b = getelementptr inbounds [64 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  %val.b = load i32, ptr addrspace(3) %gep.b, align 4
  %val = add i32 %val.a, %val.b
  store i32 %val, ptr addrspace(1) %arg, align 4
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
; CHECK: store i32 1, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, align 16, !alias.scope !6, !noalias !9
; CHECK: %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 %i
; CHECK: %val.a = load i32, ptr addrspace(3) %gep.a, align 4, !alias.scope !6, !noalias !9
; CHECK: store i32 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 1), align 16, !alias.scope !12, !noalias !13
; CHECK: %val.b = load i32, ptr addrspace(3) %gep.b, align 4, !alias.scope !12, !noalias !13
; CHECK: store i32 3, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 2), align 16, !alias.scope !14, !noalias !15
; CHECK: %val.c = load i32, ptr addrspace(3) %gep.c, align 4, !alias.scope !14, !noalias !15

define amdgpu_kernel void @no_clobber_ds_load_stores_x3(ptr addrspace(1) %arg, i32 %i) {
bb:
  store i32 1, ptr addrspace(3) @a, align 4
  %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  %val.a = load i32, ptr addrspace(3) %gep.a, align 4
  store i32 2, ptr addrspace(3) @b, align 4
  %gep.b = getelementptr inbounds [64 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  %val.b = load i32, ptr addrspace(3) %gep.b, align 4
  store i32 3, ptr addrspace(3) @c, align 4
  %gep.c = getelementptr inbounds [64 x i32], ptr addrspace(3) @c, i32 0, i32 %i
  %val.c = load i32, ptr addrspace(3) %gep.c, align 4
  %val.1 = add i32 %val.a, %val.b
  %val = add i32 %val.1, %val.c
  store i32 %val, ptr addrspace(1) %arg, align 4
  ret void
}

; CHECK: !0 = !{i64 0, i64 1}
; CHECK: !1 = !{!2}
; CHECK: !2 = distinct !{!2, !3}
; CHECK: !3 = distinct !{!3}
; CHECK: !4 = !{!5}
; CHECK: !5 = distinct !{!5, !3}
; CHECK: !6 = !{!7}
; CHECK: !7 = distinct !{!7, !8}
; CHECK: !8 = distinct !{!8}
; CHECK: !9 = !{!10, !11}
; CHECK: !10 = distinct !{!10, !8}
; CHECK: !11 = distinct !{!11, !8}
; CHECK: !12 = !{!10}
; CHECK: !13 = !{!7, !11}
; CHECK: !14 = !{!11}
; CHECK: !15 = !{!7, !10}
