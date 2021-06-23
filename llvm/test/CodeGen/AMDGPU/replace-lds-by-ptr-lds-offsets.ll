; RUN: opt -S -mtriple=amdgcn-- -amdgpu-replace-lds-use-with-pointer < %s | FileCheck --check-prefix=POINTER-REPLACE %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-replace-lds-use-with-pointer -amdgpu-lower-module-lds < %s | FileCheck --check-prefix=LOWER_LDS %s
; RUN: llc  -mtriple=amdgcn-- -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s

;
; DESCRIPTION:
;
; 1. There are three lds defined - @lds.1, @lds.2 and @lds.3, which are of types i32, i64, and [2 x i64].
;    @lds.3 is aliased to to @alias.to.lds.3
; 2. @lds.1 is used in function @f1, and @lds.2 is used in function @f2, @alias.to.lds.3 is used in kernel @k1.

; 3. Pointer-replacement pass replaces @lds.1 and @lds.2 by pointers @lds.1.ptr and @lds.2.ptr respectively.
;    However it does not touch @lds.3 since it is used in global scope (aliased).
;
; 4. LDS-lowering pass sees use of @lds.1.ptr in function @f1, use of @lds.2.ptr in function @f2, and use of
;    @lds.3 (via alias @alias.to.lds.3) in kernel @k1. Hence it module lowers these lds into struct instance
;    @llvm.amdgcn.module.lds.
;
;    The struct member order is - [lds.3, lds.1.ptr, lds.2.ptr]. Since @llvm.amdgcn.module.lds itself is allocated
;    on address 0, lds.3 is allocated on address 0, lds.1.ptr is allocated on address 16, and lds.2.ptr is allocated
;    on address 18.
;
;    Again LDS-lowering pass sees use of @lds.1 and @lds.2 in kernel. Hence it kernel lowers these lds into struct
;    instance @llvm.amdgcn.kernel.k1.lds.
;
;    The struct member order is - [@lds.2, @lds.1]. By now, already (16 + 2 + 2) 20 byte of memory allocated, @lds.2
;    is allocated on address 24 since it needs to be allocated on 8 byte boundary, and @lds.1 is allocated on address
;    32.
;
; 5. Hence the final GCN ISA looks as below:
;
;    Within kernel @k1:
;       address 24 is stored in address 18.
;       address 32 is stored in address 16
;
;    Within function @f1:
;       address 32 is loaded from address 16
;
;    Within function @f2:
;       address 24 is loaded from address 18
;


; POINTER-REPLACE: @lds.1 = addrspace(3) global i32 undef, align 4
; POINTER-REPLACE: @lds.2 = addrspace(3) global i64 undef, align 8
; POINTER-REPLACE: @lds.3 = addrspace(3) global [2 x i64] undef, align 16
; POINTER-REPLACE: @lds.1.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; POINTER-REPLACE: @lds.2.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; POINTER-REPLACE: @alias.to.lds.3 = alias [2 x i64], [2 x i64] addrspace(3)* @lds.3


; LOWER_LDS-NOT: @lds.1
; LOWER_LDS-NOT: @lds.2
; LOWER_LDS-NOT: @lds.3
; LOWER_LDS: %llvm.amdgcn.module.lds.t = type { [2 x i64], i16, i16 }
; LOWER_LDS: %llvm.amdgcn.kernel.k1.lds.t = type { i64, i32 }
; LOWER_LDS: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 16
; LOWER_LDS: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
; LOWER_LDS: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 8
; LOWER_LDS: @alias.to.lds.3 = alias [2 x i64], getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0)

@lds.1 = addrspace(3) global i32 undef, align 4
@lds.2 = addrspace(3) global i64 undef, align 8
@lds.3 = addrspace(3) global [2 x i64] undef, align 16
@alias.to.lds.3 = alias [2 x i64], [2 x i64] addrspace(3)* @lds.3

; POINTER-REPLACE-LABEL: @f1
; POINTER-REPLACE:   %1 = load i16, i16 addrspace(3)* @lds.1.ptr, align 2
; POINTER-REPLACE:   %2 = getelementptr i8, i8 addrspace(3)* null, i16 %1
; POINTER-REPLACE:   %3 = bitcast i8 addrspace(3)* %2 to i32 addrspace(3)*
; POINTER-REPLACE:   store i32 7, i32 addrspace(3)* %3, align 4
; POINTER-REPLACE:   ret void


; LOWER_LDS-LABEL: @f1
; LOWER_LDS:   %1 = load i16, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1), align 16
; LOWER_LDS:   %2 = getelementptr i8, i8 addrspace(3)* null, i16 %1
; LOWER_LDS:   %3 = bitcast i8 addrspace(3)* %2 to i32 addrspace(3)*
; LOWER_LDS:   store i32 7, i32 addrspace(3)* %3, align 4
; LOWER_LDS:   ret void


; GCN-LABEL: f1:
; GCN:         s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN:         v_mov_b32_e32 v0, 0
; GCN:         ds_read_i16 v0, v0 offset:16
; GCN:         v_mov_b32_e32 v1, 7
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         ds_write_b32 v0, v1
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_setpc_b64 s[30:31]
define void @f1() {
  store i32 7, i32 addrspace(3)* @lds.1
  ret void
}

; POINTER-REPLACE-LABEL: @f2
; POINTER-REPLACE:   %1 = load i16, i16 addrspace(3)* @lds.2.ptr, align 2
; POINTER-REPLACE:   %2 = getelementptr i8, i8 addrspace(3)* null, i16 %1
; POINTER-REPLACE:   %3 = bitcast i8 addrspace(3)* %2 to i64 addrspace(3)*
; POINTER-REPLACE:   store i64 15, i64 addrspace(3)* %3, align 4
; POINTER-REPLACE:   ret void


; LOWER_LDS-LABEL: @f2
; LOWER_LDS:   %1 = load i16, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 2), align 2
; LOWER_LDS:   %2 = getelementptr i8, i8 addrspace(3)* null, i16 %1
; LOWER_LDS:   %3 = bitcast i8 addrspace(3)* %2 to i64 addrspace(3)*
; LOWER_LDS:   store i64 15, i64 addrspace(3)* %3, align 4
; LOWER_LDS:   ret void


; GCN-LABEL: f2:
; GCN:         s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN:         v_mov_b32_e32 v1, 0
; GCN:         ds_read_i16 v2, v1 offset:18
; GCN:         v_mov_b32_e32 v0, 15
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         ds_write_b64 v2, v[0:1]
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_setpc_b64 s[30:31]
define void @f2() {
  store i64 15, i64 addrspace(3)* @lds.2
  ret void
}

; POINTER-REPLACE-LABEL: @k1
; POINTER-REPLACE:   %1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; POINTER-REPLACE:   %2 = icmp eq i32 %1, 0
; POINTER-REPLACE:   br i1 %2, label %3, label %4
;
; POINTER-REPLACE-LABEL: 3:
; POINTER-REPLACE:   store i16 ptrtoint (i64 addrspace(3)* @lds.2 to i16), i16 addrspace(3)* @lds.2.ptr, align 2
; POINTER-REPLACE:   store i16 ptrtoint (i32 addrspace(3)* @lds.1 to i16), i16 addrspace(3)* @lds.1.ptr, align 2
; POINTER-REPLACE:   br label %4
;
; POINTER-REPLACE-LABEL: 4:
; POINTER-REPLACE:   call void @llvm.amdgcn.wave.barrier()
; POINTER-REPLACE:   %bc = bitcast [2 x i64] addrspace(3)* @alias.to.lds.3 to i8 addrspace(3)*
; POINTER-REPLACE:   store i8 3, i8 addrspace(3)* %bc, align 2
; POINTER-REPLACE:   call void @f1()
; POINTER-REPLACE:   call void @f2()
; POINTER-REPLACE:   ret void


; LOWER_LDS-LABEL: @k1
; LOWER_LDS:   call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; LOWER_LDS:   %1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; LOWER_LDS:   %2 = icmp eq i32 %1, 0
; LOWER_LDS:   br i1 %2, label %3, label %6
;
; LOWER_LDS-LABEL: 3:
; LOWER_LDS:   %4 = ptrtoint i64 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 0) to i16
; LOWER_LDS:   store i16 %4, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 2), align 2
; LOWER_LDS:   %5 = ptrtoint i32 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1) to i16
; LOWER_LDS:   store i16 %5, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1), align 16
; LOWER_LDS:   br label %6
;
; LOWER_LDS-LABEL: 6:
; LOWER_LDS:   call void @llvm.amdgcn.wave.barrier()
; LOWER_LDS:   %bc = bitcast [2 x i64] addrspace(3)* @alias.to.lds.3 to i8 addrspace(3)*
; LOWER_LDS:   store i8 3, i8 addrspace(3)* %bc, align 2
; LOWER_LDS:   call void @f1()
; LOWER_LDS:   call void @f2()
; LOWER_LDS:   ret void


; GCN-LABEL: k1:
; GCN:         s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; GCN:         s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; GCN:         s_mov_b32 s10, -1
; GCN:         s_mov_b32 s11, 0xe00000
; GCN:         s_add_u32 s8, s8, s1
; GCN:         v_mbcnt_lo_u32_b32 v0, -1, 0
; GCN:         s_addc_u32 s9, s9, 0
; GCN:         v_cmp_eq_u32_e32 vcc, 0, v0
; GCN:         s_mov_b32 s32, 0
; GCN:         s_and_saveexec_b64 s[0:1], vcc
; GCN:         s_cbranch_execz BB2_2
; GCN:         v_mov_b32_e32 v0, 0
; GCN:         v_mov_b32_e32 v1, 0x180020
; GCN:         ds_write_b32 v0, v1 offset:16
; GCN-LABEL: BB2_2:
; GCN:         s_or_b64 exec, exec, s[0:1]
; GCN:         s_getpc_b64 s[0:1]
; GCN:         s_add_u32 s0, s0, f1@gotpcrel32@lo+4
; GCN:         s_addc_u32 s1, s1, f1@gotpcrel32@hi+12
; GCN:         s_load_dwordx2 s[4:5], s[0:1], 0x0
; GCN:         s_mov_b64 s[0:1], s[8:9]
; GCN:         s_mov_b64 s[2:3], s[10:11]
; GCN:         v_mov_b32_e32 v0, alias.to.lds.3@abs32@lo
; GCN:         v_mov_b32_e32 v1, 3
;              ; wave barrier
; GCN:         ds_write_b8 v0, v1
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_swappc_b64 s[30:31], s[4:5]
; GCN:         s_getpc_b64 s[0:1]
; GCN:         s_add_u32 s0, s0, f2@gotpcrel32@lo+4
; GCN:         s_addc_u32 s1, s1, f2@gotpcrel32@hi+12
; GCN:         s_load_dwordx2 s[4:5], s[0:1], 0x0
; GCN:         s_mov_b64 s[0:1], s[8:9]
; GCN:         s_mov_b64 s[2:3], s[10:11]
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_swappc_b64 s[30:31], s[4:5]
; GCN:         s_endpgm
define amdgpu_kernel void @k1() {
  %bc = bitcast [2 x i64] addrspace(3)* @alias.to.lds.3 to i8 addrspace(3)*
  store i8 3, i8 addrspace(3)* %bc, align 2
  call void @f1()
  call void @f2()
  ret void
}
