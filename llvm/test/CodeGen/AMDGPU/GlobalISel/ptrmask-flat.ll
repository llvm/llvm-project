; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -O2 -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -print-after=amdgpu-reg-bank-legalize,si-fold-operands < %s -o %t.s 2> %t.mir
; RUN: FileCheck %s --check-prefixes=BANK,SEL,COPIED < %t.mir || FileCheck %s --check-prefixes=BANK,SEL,DIRECT < %t.mir
; RUN: FileCheck %s --check-prefixes=ASM,PACKED -DX=v0 -D#OUT=2 < %t.s || \
; RUN:   FileCheck %s --check-prefixes=ASM,SPLIT -DX=v0 -D#OUT=0 < %t.s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -O2 -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -print-after=amdgpu-reg-bank-legalize,si-fold-operands < %s -o %t.s 2> %t.mir
; RUN: FileCheck %s --check-prefixes=BANK,SEL,COPIED < %t.mir || FileCheck %s --check-prefixes=BANK,SEL,DIRECT < %t.mir
; RUN: FileCheck %s --check-prefixes=ASM,GFX11,PACKED -D#OUT=2 < %t.s || \
; RUN:   FileCheck %s --check-prefixes=ASM,GFX11,SPLIT -D#OUT=0 < %t.s

define amdgpu_kernel void @flat_uu(ptr %ptr, i64 %mask, ptr addrspace(1) %out) {
; BANK-LABEL: Machine code for function flat_uu:
; BANK: [[ARGS:%[0-9]+]]:sgpr(<2 x i64>) = G_LOAD {{.*}} :: ({{.*}}from %ir.ptr.kernarg.offset{{[0-9]*}},
; BANK: [[BITS:%[0-9]+]]:sgpr(i64), [[UMASK:%[0-9]+]]:sgpr(i64) = G_UNMERGE_VALUES [[ARGS]]:
; BANK: [[UPTR:%[0-9]+]]:sgpr(p0) = G_INTTOPTR [[BITS]]:sgpr(i64)
; BANK: [[MASKED:%[0-9]+]]:sgpr(p0) = G_PTRMASK [[UPTR]]:sgpr, [[UMASK]]:sgpr(i64)
; BANK: [[STORED:%[0-9]+]]:vgpr(p0) = COPY [[MASKED]]:sgpr(p0)
; BANK: G_STORE [[STORED]]:vgpr(p0), {{%[0-9]+}}:vgpr(p1) :: (store (p0) into %ir.dest,
;
; ASM-LABEL: flat_uu:
; ASM: s_and_b64
; ASM: global_store_{{dwordx2|b64}}
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %dest = getelementptr ptr, ptr addrspace(1) %out, i32 %id
  %result = call ptr @llvm.ptrmask.p0.i64(ptr %ptr, i64 %mask)
  store ptr %result, ptr addrspace(1) %dest, align 8
  ret void
}

define amdgpu_kernel void @flat_du(ptr addrspace(1) %src, ptr addrspace(1) %out, i64 %mask) {
; BANK-LABEL: Machine code for function flat_du:
; BANK: [[PTR:%[0-9]+]]:vgpr(p0) = G_LOAD {{.*}} :: (load (p0) from %ir.slot,
; The uniform mask may be copied to a VGPR or consumed directly from an SGPR.
; BANK: [[MASKED:%[0-9]+]]:vgpr(p0) = G_PTRMASK [[PTR]]:vgpr, {{%[0-9]+:[sv]gpr}}(i64)
; BANK: G_STORE [[MASKED]]:vgpr(p0), {{%[0-9]+}}:vgpr(p1) :: (store (p0) into %ir.dest,
;
; After operand folding, both inputs to each AND name their 32-bit subregisters.
; SEL: IR Dump After SI Fold Operands
; The mask kernel argument is at byte 16. Operand folding can eliminate its VGPR copy.
; COPIED: [[ARG:%[0-9]+]]:{{[^ ]+}} = S_LOAD_{{[^ ]+}} {{.*}} :: ({{.*}}load (i64) from %ir.src.kernarg.offset{{[0-9]*}} + 16,
; COPIED: [[MASK:%[0-9]+]]:vreg_64 = COPY [[ARG]]:
; DIRECT: [[MASK:%[0-9]+]]:{{[^ ]+}} = S_LOAD_{{[^ ]+}} {{.*}} :: ({{.*}}load (i64) from %ir.src.kernarg.offset{{[0-9]*}} + 16,
; SEL-DAG: [[LO:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 [[PTR]].sub0:vreg_64, [[MASK]].sub0:
; SEL-DAG: [[HI:%[0-9]+]]:vgpr_32 = V_AND_B32_e64 [[PTR]].sub1:vreg_64, [[MASK]].sub1:
; SEL: [[MASKED]]:vreg_64 = REG_SEQUENCE [[LO]]:vgpr_32, %subreg.sub0, [[HI]]:vgpr_32, %subreg.sub1
; SEL: GLOBAL_STORE_{{[^ ]+}} {{[^,]+}}, [[MASKED]]:vreg_64, {{.*}} :: (store (p0) into %ir.dest,
;
; ASM-LABEL: flat_du:
; out is the high pair of a combined argument load or the result of a load at byte 8.
; GFX11-DAG: {{v_(dual_)?and_b32(_e32)?}} [[X:v[0-9]+]], 0x3ff, v0{{( |$)}}
; ASM-DAG: {{v_(dual_)?lshlrev_b32(_e32)?}} [[OFFSET:v[0-9]+]], 3, [[X]]{{( |$)}}
; PACKED-DAG: s_load_{{dwordx4|b128}} s{{\[}}[[#BASE:]]:[[#]]], {{.*}}, 0x0{{$}}
; SPLIT-DAG: s_load_{{dwordx2|b64}} s{{\[}}[[#BASE:]]:[[#]]], {{.*}}, 0x8{{$}}
; ASM: global_load_{{dwordx2|b64}} v{{\[}}[[#P:]]:[[#]]], [[OFFSET]],
; ASM-DAG: {{v_(dual_)?and_b32(_e(32|64))?}} v[[#LO:]], {{([sv][0-9]+, )?}}v[[#P]]{{(, [sv][0-9]+)?( |$)}}
; ASM-DAG: {{v_(dual_)?and_b32(_e(32|64))?}} v[[#HI:]], {{([sv][0-9]+, )?}}v[[#P+1]]{{(, [sv][0-9]+)?( |$)}}
; ASM: global_store_{{dwordx2|b64}} [[OFFSET]], v{{\[}}[[#LO]]:[[#HI]]], s{{\[}}[[#BASE+OUT]]:[[#BASE+OUT+1]]]{{$}}
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %slot = getelementptr ptr, ptr addrspace(1) %src, i32 %id
  %dest = getelementptr ptr, ptr addrspace(1) %out, i32 %id
  %ptr = load ptr, ptr addrspace(1) %slot, align 8
  %result = call ptr @llvm.ptrmask.p0.i64(ptr %ptr, i64 %mask)
  store ptr %result, ptr addrspace(1) %dest, align 8
  ret void
}
