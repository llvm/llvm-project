; RUN: opt -S -mtriple=amdgcn--amdhsa -passes=amdgpu-lower-module-lds < %s --amdgpu-lower-module-lds-strategy=table | FileCheck -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s --amdgpu-lower-module-lds-strategy=table | FileCheck -check-prefix=GCN %s

; Opt checks from utils/update_test_checks.py, llc checks from utils/update_llc_test_checks.py, both modified.

; Define four variables and four non-kernel functions which access exactly one variable each
@v0 = addrspace(3) global float undef
@v1 = addrspace(3) global i16 undef, align 16
@v2 = addrspace(3) global i64 undef
@v3 = addrspace(3) global i8 undef
@unused = addrspace(3) global i16 undef

; OPT: %llvm.amdgcn.kernel.kernel_no_table.lds.t = type { i64 }
; OPT: %llvm.amdgcn.kernel.k01.lds.t = type { i16, [2 x i8], float }
; OPT: %llvm.amdgcn.kernel.k23.lds.t = type { i64, i8 }
; OPT: %llvm.amdgcn.kernel.k123.lds.t = type { i16, i8, [5 x i8], i64 }

; OPT: @llvm.amdgcn.kernel.kernel_no_table.lds = internal addrspace(3) global %llvm.amdgcn.kernel.kernel_no_table.lds.t undef, align 8, !absolute_symbol !0
; OPT: @llvm.amdgcn.kernel.k01.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k01.lds.t undef, align 16, !absolute_symbol !0
; OPT: @llvm.amdgcn.kernel.k23.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k23.lds.t undef, align 8, !absolute_symbol !0
; OPT: @llvm.amdgcn.kernel.k123.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k123.lds.t undef, align 16, !absolute_symbol !0

; Salient parts of the IR lookup table check:
; It has (top level) size 3 as there are 3 kernels that call functions which use lds
; The next level down has type [4 x i16] as there are 4 variables accessed by functions which use lds
; The kernel naming pattern and the structs being named after the functions helps verify placement of undef
; The remainder are constant expressions into the variable instances checked above

; OPT{LITERAL}: @llvm.amdgcn.lds.offset.table = internal addrspace(4) constant [3 x [4 x i32]] [[4 x i32] [i32 ptrtoint (ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k01.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k01.lds, i32 0, i32 2) to i32), i32 ptrtoint (ptr addrspace(3) @llvm.amdgcn.kernel.k01.lds to i32), i32 poison, i32 poison], [4 x i32] [i32 poison, i32 ptrtoint (ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k123.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds, i32 0, i32 3) to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k123.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds, i32 0, i32 1) to i32)], [4 x i32] [i32 poison, i32 poison, i32 ptrtoint (ptr addrspace(3) @llvm.amdgcn.kernel.k23.lds to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k23.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k23.lds, i32 0, i32 1) to i32)]]


define void @f0() {
; OPT-LABEL: @f0(
; OPT-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.lds.kernel.id()
; OPT-NEXT:    [[V02:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 0
; OPT-NEXT:    [[TMP2:%.*]] = load i32, ptr addrspace(4) [[V02]], align 4
; OPT-NEXT:    [[V03:%.*]] = inttoptr i32 [[TMP2]] to ptr addrspace(3)
; OPT-NEXT:    [[LD:%.*]] = load float, ptr addrspace(3) [[V03]], align 4
; OPT-NEXT:    [[MUL:%.*]] = fmul float [[LD]], 2.000000e+00
; OPT-NEXT:    [[V0:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 0
; OPT-NEXT:    [[TMP3:%.*]] = load i32, ptr addrspace(4) [[V0]], align 4
; OPT-NEXT:    [[V01:%.*]] = inttoptr i32 [[TMP3]] to ptr addrspace(3)
; OPT-NEXT:    store float [[MUL]], ptr addrspace(3) [[V01]], align 4
; OPT-NEXT:    ret void
;
; GCN-LABEL: f0:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s15
; GCN-NEXT:    s_ashr_i32 s5, s15, 31
; GCN-NEXT:    s_getpc_b64 s[6:7]
; GCN-NEXT:    s_add_u32 s6, s6, llvm.amdgcn.lds.offset.table@rel32@lo+4
; GCN-NEXT:    s_addc_u32 s7, s7, llvm.amdgcn.lds.offset.table@rel32@hi+12
; GCN-NEXT:    s_lshl_b64 s[4:5], s[4:5], 4
; GCN-NEXT:    s_add_u32 s4, s4, s6
; GCN-NEXT:    s_addc_u32 s5, s5, s7
; GCN-NEXT:    s_load_dword s4, s[4:5], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_b32 v1, v0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_add_f32_e32 v1, v1, v1
; GCN-NEXT:    ds_write_b32 v0, v1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %ld = load float, ptr addrspace(3) @v0
  %mul = fmul float %ld, 2.
  store float %mul, ptr  addrspace(3) @v0
  ret void
}

define void @f1() {
; OPT-LABEL: @f1(
; OPT-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.lds.kernel.id()
; OPT-NEXT:    [[V12:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 1
; OPT-NEXT:    [[TMP2:%.*]] = load i32, ptr addrspace(4) [[V12]], align 4
; OPT-NEXT:    [[V13:%.*]] = inttoptr i32 [[TMP2]] to ptr addrspace(3)
; OPT-NEXT:    [[LD:%.*]] = load i16, ptr addrspace(3) [[V13]], align 2
; OPT-NEXT:    [[MUL:%.*]] = mul i16 [[LD]], 3
; OPT-NEXT:    [[V1:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 1
; OPT-NEXT:    [[TMP3:%.*]] = load i32, ptr addrspace(4) [[V1]], align 4
; OPT-NEXT:    [[V11:%.*]] = inttoptr i32 [[TMP3]] to ptr addrspace(3)
; OPT-NEXT:    store i16 [[MUL]], ptr addrspace(3) [[V11]], align 2
; OPT-NEXT:    ret void
;
; GCN-LABEL: f1:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s15
; GCN-NEXT:    s_ashr_i32 s5, s15, 31
; GCN-NEXT:    s_getpc_b64 s[6:7]
; GCN-NEXT:    s_add_u32 s6, s6, llvm.amdgcn.lds.offset.table@rel32@lo+8
; GCN-NEXT:    s_addc_u32 s7, s7, llvm.amdgcn.lds.offset.table@rel32@hi+16
; GCN-NEXT:    s_lshl_b64 s[4:5], s[4:5], 4
; GCN-NEXT:    s_add_u32 s4, s4, s6
; GCN-NEXT:    s_addc_u32 s5, s5, s7
; GCN-NEXT:    s_load_dword s4, s[4:5], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_u16 v1, v0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mul_lo_u32 v1, v1, 3
; GCN-NEXT:    ds_write_b16 v0, v1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %ld = load i16, ptr addrspace(3) @v1
  %mul = mul i16 %ld, 3
  store i16 %mul, ptr  addrspace(3) @v1
  ret void
}

define void @f2() {
; OPT-LABEL: @f2(
; OPT-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.lds.kernel.id()
; OPT-NEXT:    [[V22:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 2
; OPT-NEXT:    [[TMP2:%.*]] = load i32, ptr addrspace(4) [[V22]], align 4
; OPT-NEXT:    [[V23:%.*]] = inttoptr i32 [[TMP2]] to ptr addrspace(3)
; OPT-NEXT:    [[LD:%.*]] = load i64, ptr addrspace(3) [[V23]], align 4
; OPT-NEXT:    [[MUL:%.*]] = mul i64 [[LD]], 4
; OPT-NEXT:    [[V2:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 2
; OPT-NEXT:    [[TMP3:%.*]] = load i32, ptr addrspace(4) [[V2]], align 4
; OPT-NEXT:    [[V21:%.*]] = inttoptr i32 [[TMP3]] to ptr addrspace(3)
; OPT-NEXT:    store i64 [[MUL]], ptr addrspace(3) [[V21]], align 4
; OPT-NEXT:    ret void
;
; GCN-LABEL: f2:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s15
; GCN-NEXT:    s_ashr_i32 s5, s15, 31
; GCN-NEXT:    s_getpc_b64 s[6:7]
; GCN-NEXT:    s_add_u32 s6, s6, llvm.amdgcn.lds.offset.table@rel32@lo+12
; GCN-NEXT:    s_addc_u32 s7, s7, llvm.amdgcn.lds.offset.table@rel32@hi+20
; GCN-NEXT:    s_lshl_b64 s[4:5], s[4:5], 4
; GCN-NEXT:    s_add_u32 s4, s4, s6
; GCN-NEXT:    s_addc_u32 s5, s5, s7
; GCN-NEXT:    s_load_dword s4, s[4:5], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v2, s4
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_b64 v[0:1], v2
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_lshl_b64 v[0:1], v[0:1], 2
; GCN-NEXT:    ds_write_b64 v2, v[0:1]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %ld = load i64, ptr addrspace(3) @v2
  %mul = mul i64 %ld, 4
  store i64 %mul, ptr  addrspace(3) @v2
  ret void
}

define void @f3() {
; OPT-LABEL: @f3(
; OPT-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.lds.kernel.id()
; OPT-NEXT:    [[V32:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 3
; OPT-NEXT:    [[TMP2:%.*]] = load i32, ptr addrspace(4) [[V32]], align 4
; OPT-NEXT:    [[V33:%.*]] = inttoptr i32 [[TMP2]] to ptr addrspace(3)
; OPT-NEXT:    [[LD:%.*]] = load i8, ptr addrspace(3) [[V33]], align 1
; OPT-NEXT:    [[MUL:%.*]] = mul i8 [[LD]], 5
; OPT-NEXT:    [[V3:%.*]] = getelementptr inbounds [3 x [4 x i32]], ptr addrspace(4) @llvm.amdgcn.lds.offset.table, i32 0, i32 [[TMP1]], i32 3
; OPT-NEXT:    [[TMP3:%.*]] = load i32, ptr addrspace(4) [[V3]], align 4
; OPT-NEXT:    [[V31:%.*]] = inttoptr i32 [[TMP3]] to ptr addrspace(3)
; OPT-NEXT:    store i8 [[MUL]], ptr addrspace(3) [[V31]], align 1
; OPT-NEXT:    ret void
;
; GCN-LABEL: f3:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s15
; GCN-NEXT:    s_ashr_i32 s5, s15, 31
; GCN-NEXT:    s_getpc_b64 s[6:7]
; GCN-NEXT:    s_add_u32 s6, s6, llvm.amdgcn.lds.offset.table@rel32@lo+16
; GCN-NEXT:    s_addc_u32 s7, s7, llvm.amdgcn.lds.offset.table@rel32@hi+24
; GCN-NEXT:    s_lshl_b64 s[4:5], s[4:5], 4
; GCN-NEXT:    s_add_u32 s4, s4, s6
; GCN-NEXT:    s_addc_u32 s5, s5, s7
; GCN-NEXT:    s_load_dword s4, s[4:5], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_u8 v1, v0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mul_lo_u32 v1, v1, 5
; GCN-NEXT:    ds_write_b8 v0, v1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %ld = load i8, ptr addrspace(3) @v3
  %mul = mul i8 %ld, 5
  store i8 %mul, ptr  addrspace(3) @v3
  ret void
}

; Doesn't access any via a function, won't be in the lookup table
define amdgpu_kernel void @kernel_no_table() {
; OPT-LABEL: @kernel_no_table() {
; OPT-NEXT:    [[LD:%.*]] = load i64, ptr addrspace(3) @llvm.amdgcn.kernel.kernel_no_table.lds, align 8
; OPT-NEXT:    [[MUL:%.*]] = mul i64 [[LD]], 8
; OPT-NEXT:    store i64 [[MUL]], ptr addrspace(3) @llvm.amdgcn.kernel.kernel_no_table.lds, align 8
; OPT-NEXT:    ret void
;
; GCN-LABEL: kernel_no_table:
; GCN:       ; %bb.0:
; GCN-NEXT:    v_mov_b32_e32 v2, 0
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_b64 v[0:1], v2
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_lshl_b64 v[0:1], v[0:1], 3
; GCN-NEXT:    ds_write_b64 v2, v[0:1]
; GCN-NEXT:    s_endpgm
  %ld = load i64, ptr addrspace(3) @v2
  %mul = mul i64 %ld, 8
  store i64 %mul, ptr  addrspace(3) @v2
  ret void
}

; Access two variables, will allocate those two
define amdgpu_kernel void @k01() {
; OPT-LABEL: @k01() !llvm.amdgcn.lds.kernel.id !1 {
; OPT-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(ptr addrspace(3) @llvm.amdgcn.kernel.k01.lds) ]
; OPT-NEXT:    call void @f0()
; OPT-NEXT:    call void @f1()
; OPT-NEXT:    ret void
;
; GCN-LABEL: k01:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_mov_b32 s32, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s7
; GCN-NEXT:    s_add_i32 s6, s6, s9
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s6, 8
; GCN-NEXT:    s_add_u32 s0, s0, s9
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b64 s[8:9], s[4:5]
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f0@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f0@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_mov_b32 s15, 0
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f1@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f1@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    s_endpgm
; GCN:         .amdhsa_group_segment_fixed_size 8
  call void @f0()
  call void @f1()
  ret void
}

define amdgpu_kernel void @k23() {
; OPT-LABEL: @k23() !llvm.amdgcn.lds.kernel.id !2 {
; OPT-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(ptr addrspace(3) @llvm.amdgcn.kernel.k23.lds) ]
; OPT-NEXT:    call void @f2()
; OPT-NEXT:    call void @f3()
; OPT-NEXT:    ret void
;
; GCN-LABEL: k23:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_mov_b32 s32, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s7
; GCN-NEXT:    s_add_i32 s6, s6, s9
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s6, 8
; GCN-NEXT:    s_add_u32 s0, s0, s9
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b64 s[8:9], s[4:5]
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f2@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f2@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_mov_b32 s15, 2
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f3@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f3@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    s_endpgm
; GCN:         .amdhsa_group_segment_fixed_size 16
  call void @f2()
  call void @f3()
  ret void
}

; Access and allocate three variables
define amdgpu_kernel void @k123() {
; OPT-LABEL: @k123() !llvm.amdgcn.lds.kernel.id !3 {
; OPT-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds) ]
; OPT-NEXT:    call void @f1()
; OPT-NEXT:    [[LD:%.*]] = load i8, ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_K123_LDS_T:%.*]], ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds, i32 0, i32 1), align 2, !alias.scope !4, !noalias !7
; OPT-NEXT:    [[MUL:%.*]] = mul i8 [[LD]], 8
; OPT-NEXT:    store i8 [[MUL]], ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_K123_LDS_T]], ptr addrspace(3) @llvm.amdgcn.kernel.k123.lds, i32 0, i32 1), align 2, !alias.scope !4, !noalias !7
; OPT-NEXT:    call void @f2()
; OPT-NEXT:    ret void
;
; GCN-LABEL: k123:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_mov_b32 s32, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s7
; GCN-NEXT:    s_add_i32 s6, s6, s9
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s6, 8
; GCN-NEXT:    s_add_u32 s0, s0, s9
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b64 s[8:9], s[4:5]
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f1@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f1@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_mov_b32 s15, 1
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_read_u8 v1, v0 offset:2
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:    s_add_u32 s4, s4, f2@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s5, s5, f2@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_lshlrev_b32_e32 v1, 3, v1
; GCN-NEXT:    ds_write_b8 v0, v1 offset:2
; GCN-NEXT:    s_mov_b64 s[6:7], s[8:9]
; GCN-NEXT:    s_swappc_b64 s[30:31], s[4:5]
; GCN-NEXT:    s_endpgm
; GCN:         .amdhsa_group_segment_fixed_size 16
  call void @f1()
  %ld = load i8, ptr addrspace(3) @v3
  %mul = mul i8 %ld, 8
  store i8 %mul, ptr  addrspace(3) @v3
  call void @f2()
  ret void
}


; OPT: declare i32 @llvm.amdgcn.lds.kernel.id()

!0 = !{i64 0, i64 1}
!1 = !{i32 0}
!2 = !{i32 2}
!3 = !{i32 1}


; Table size length number-kernels * number-variables * sizeof(uint16_t)
; GCN:      .type	llvm.amdgcn.lds.offset.table,@object
; GCN-NEXT: .section	.data.rel.ro,#alloc,#write
; GCN-NEXT: .p2align	4, 0x0
; GCN-NEXT: llvm.amdgcn.lds.offset.table:
; GCN-NEXT: .long	0+4
; GCN-NEXT: .long	0
; GCN-NEXT: .zero	4
; GCN-NEXT: .zero	4
; GCN-NEXT: .zero	4
; GCN-NEXT: .long	0
; GCN-NEXT: .long	0+8
; GCN-NEXT: .long	0+2
; GCN-NEXT: .zero	4
; GCN-NEXT: .zero	4
; GCN-NEXT: .long	0
; GCN-NEXT: .long	0+8
; GCN-NEXT: .size	llvm.amdgcn.lds.offset.table, 48
