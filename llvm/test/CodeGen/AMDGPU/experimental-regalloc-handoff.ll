; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -stop-after=finalize-isel \
; RUN:   -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=DAG
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -stop-after=finalize-isel \
; RUN:   -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=GFX9
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -global-isel \
; RUN:   -global-isel-abort=1 -stop-after=irtranslator -o - %s \
; RUN:   | FileCheck %s --check-prefix=GISEL-IR
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -global-isel \
; RUN:   -global-isel-abort=1 -stop-after=amdgpu-reg-bank-select \
; RUN:   -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=GISEL-RBS
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O0 -global-isel \
; RUN:   -global-isel-abort=1 -verify-machineinstrs -o - %s \
; RUN:   | FileCheck %s --check-prefix=FULL
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -global-isel \
; RUN:   -global-isel-abort=1 -verify-machineinstrs -o - %s \
; RUN:   | FileCheck %s --check-prefix=FULL
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O0 \
; RUN:   -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=FULL
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 \
; RUN:   -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=FULL

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

define amdgpu_kernel void @handoff_vgpr(ptr addrspace(1) %out) {
; DAG-LABEL: name: handoff_vgpr
; DAG: [[SRC:%[0-9]+]]:av_32 = COPY
; DAG-NEXT: [[HANDOFF:%[0-9]+]]:vgpr_32 = nomerge REGALLOC_HANDOFF_VGPR {{(killed )?}}[[SRC]]
;
; GISEL-IR-LABEL: name: handoff_vgpr
; GISEL-IR: %{{[0-9]+}}:_(i32) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.experimental.regalloc.handoff), %{{[0-9]+}}(i32), !{{[0-9]+}}
;
; GISEL-RBS-LABEL: name: handoff_vgpr
; GISEL-RBS: [[GSRC:%[0-9]+]]:av_32{{\((s32|i32)\)}} = COPY
; GISEL-RBS-NEXT: [[GHANDOFF:%[0-9]+]]:vgpr_32{{\((s32|i32)\)}} = nomerge REGALLOC_HANDOFF_VGPR [[GSRC]]
  %src = call i32 @llvm.amdgcn.workitem.id.x()
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !0)
  store i32 %dst, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @handoff_agpr(ptr addrspace(1) %out) {
; DAG-LABEL: name: handoff_agpr
; DAG: [[SRC:%[0-9]+]]:av_32 = COPY
; DAG-NEXT: [[HANDOFF:%[0-9]+]]:agpr_32 = nomerge REGALLOC_HANDOFF_AGPR {{(killed )?}}[[SRC]]
;
; GISEL-IR-LABEL: name: handoff_agpr
; GISEL-IR: %{{[0-9]+}}:_(i32) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.experimental.regalloc.handoff), %{{[0-9]+}}(i32), !{{[0-9]+}}
;
; GISEL-RBS-LABEL: name: handoff_agpr
; GISEL-RBS: [[GSRC:%[0-9]+]]:av_32{{\((s32|i32)\)}} = COPY
; GISEL-RBS-NEXT: [[GHANDOFF:%[0-9]+]]:agpr_32{{\((s32|i32)\)}} = nomerge REGALLOC_HANDOFF_AGPR [[GSRC]]
;
; GFX9-LABEL: name: handoff_agpr
; GFX9-NOT: REGALLOC_HANDOFF_
; GFX9: S_ENDPGM
  %src = call i32 @llvm.amdgcn.workitem.id.x()
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !1)
  store i32 %dst, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_vgpr(i32 %src) {
; DAG-LABEL: name: uniform_vgpr
; DAG: [[SRC:%[0-9]+]]:av_32 = COPY
; DAG-NEXT: [[HANDOFF:%[0-9]+]]:vgpr_32 = nomerge REGALLOC_HANDOFF_VGPR {{(killed )?}}[[SRC]]
; DAG-NEXT: [[SCALAR:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 {{(killed )?}}[[HANDOFF]], implicit $exec
;
; FULL-LABEL: uniform_vgpr:
; FULL: s_endpgm
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !0)
  call void asm sideeffect "; use $0", "s"(i32 %dst)
  ret void
}

define amdgpu_kernel void @uniform_agpr(i32 %src) {
; DAG-LABEL: name: uniform_agpr
; DAG: [[SRC:%[0-9]+]]:av_32 = COPY
; DAG-NEXT: [[HANDOFF:%[0-9]+]]:agpr_32 = nomerge REGALLOC_HANDOFF_AGPR {{(killed )?}}[[SRC]]
; DAG-NEXT: [[VECTOR:%[0-9]+]]:vgpr_32 = COPY {{(killed )?}}[[HANDOFF]]
; DAG-NEXT: [[SCALAR:%[0-9]+]]:sreg_32_xm0 = V_READFIRSTLANE_B32 {{(killed )?}}[[VECTOR]], implicit $exec
;
; FULL-LABEL: uniform_agpr:
; FULL: s_endpgm
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !1)
  call void asm sideeffect "; use $0", "s"(i32 %dst)
  ret void
}

define amdgpu_kernel void @uniform_consumer_vgpr(i32 %src) {
; FULL-LABEL: uniform_consumer_vgpr:
; FULL: s_endpgm
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !0)
  %sum = add i32 %dst, 1
  call void asm sideeffect "; use $0", "s"(i32 %sum)
  ret void
}

define amdgpu_kernel void @uniform_consumer_agpr(i32 %src) {
; FULL-LABEL: uniform_consumer_agpr:
; FULL: s_endpgm
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !1)
  %sum = add i32 %dst, 1
  call void asm sideeffect "; use $0", "s"(i32 %sum)
  ret void
}

define amdgpu_kernel void @phi_forward_vgpr(i32 %src, i32 %cond) {
; FULL-LABEL: phi_forward_vgpr:
; FULL: s_endpgm
entry:
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !0)
  %cmp = icmp ne i32 %cond, 0
  br i1 %cmp, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  %value = phi i32 [ %dst, %left ], [ 0, %right ]
  call void asm sideeffect "; use $0", "s"(i32 %value)
  ret void
}

define amdgpu_kernel void @phi_forward_agpr(i32 %src, i32 %cond) {
; FULL-LABEL: phi_forward_agpr:
; FULL: s_endpgm
entry:
  %dst = call i32 @llvm.experimental.regalloc.handoff(
      i32 %src, metadata !1)
  %cmp = icmp ne i32 %cond, 0
  br i1 %cmp, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  %value = phi i32 [ %dst, %left ], [ 0, %right ]
  call void asm sideeffect "; use $0", "s"(i32 %value)
  ret void
}

!0 = !{!"amdgpu.vgpr"}
!1 = !{!"amdgpu.agpr"}
