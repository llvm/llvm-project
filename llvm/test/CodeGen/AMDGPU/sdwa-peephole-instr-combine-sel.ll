; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -o - < %s | FileCheck -check-prefix=CHECK %s

; CHECK-NOT: v_lshlrev_b32_sdwa v{{[0-9]}}, v{{[0-9]}}, v{{[0-9]}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel(ptr addrspace(1) %input.coerce, i32 %0, i1 %cmp3.i, i32 %add5.1, ptr addrspace(3) %1, ptr addrspace(3) %2) {
; CHECK-LABEL: kernel:
; CHECK-NEXT:       ; %bb.0: ; %entry
; CHECK-NEXT:    s_load_dwordx4 s[0:3], s[8:9], 0x0
; CHECK-NEXT:    v_mov_b32_e32 v2, 8
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_clause 0x1
; CHECK-NEXT:    global_load_ushort v1, v0, s[0:1]
; CHECK-NEXT:    global_load_ubyte v0, v0, s[0:1] offset:2
; CHECK-NEXT:    s_bitcmp1_b32 s3, 0
; CHECK-NEXT:    s_cselect_b32 s3, -1, 0
; CHECK-NEXT:    s_and_b32 vcc_lo, exec_lo, s3
; CHECK-NEXT:    s_waitcnt vmcnt(1)
; CHECK-NEXT:    v_lshrrev_b32_sdwa v2, v2, v1 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; CHECK-NEXT:    v_or_b32_sdwa v1, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; CHECK-NEXT:    v_and_b32_e32 v1, 0xffff, v1
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_lshl_or_b32 v0, v0, 16, v1
; CHECK-NEXT:    s_cbranch_vccz .LBB0_2
; CHECK-NEXT:  ; %bb.1: ; %if.then.i
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    ds_write_b32 v1, v1
; CHECK-NEXT:  .LBB0_2: ; %if.end.i
; CHECK-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; CHECK-NEXT:    s_mov_b32 s3, exec_lo
; CHECK-NEXT:    v_cmpx_ne_u16_e32 0, v1
; CHECK-NEXT:    s_xor_b32 s3, exec_lo, s3
; CHECK-NEXT:    s_cbranch_execz .LBB0_4
; CHECK-NEXT:  ; %bb.3: ; %if.then.i.i.i.i.i
; CHECK-NEXT:    v_mov_b32_e32 v2, 2
; CHECK-NEXT:    v_lshlrev_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; CHECK-NEXT:    v_mov_b32_e32 v2, s2
; CHECK-NEXT:    ds_write_b32 v1, v2 offset:84

entry:
  %3 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %3 to i64
  %arrayidx = getelementptr i8, ptr addrspace(1) %input.coerce, i64 %idxprom
  %4 = load i8, ptr addrspace(1) %arrayidx, align 1
  %add5.11 = or disjoint i32 %3, 1
  %idxprom.1 = zext i32 %add5.11 to i64
  %arrayidx.1 = getelementptr i8, ptr addrspace(1) %input.coerce, i64 %idxprom.1
  %5 = load i8, ptr addrspace(1) %arrayidx.1, align 1
  %add5.2 = or disjoint i32 %3, 2
  %idxprom.2 = zext i32 %add5.2 to i64
  %arrayidx.2 = getelementptr i8, ptr addrspace(1) %input.coerce, i64 %idxprom.2
  %6 = load i8, ptr addrspace(1) %arrayidx.2, align 1
  br i1 %cmp3.i, label %if.then.i, label %if.end.i

if.then.i.i.i.i.i:                                ; preds = %if.end.i
  %7 = zext i8 %6 to i32
  %arrayidx7.i.i.i.i.i = getelementptr nusw [14 x i32], ptr addrspace(3) inttoptr (i32 84 to ptr addrspace(3)), i32 0, i32 %7
  store i32 %0, ptr addrspace(3) %arrayidx7.i.i.i.i.i, align 4
  br label %func.exit.i.i.i

func.exit.i.i.i: ; preds = %if.end.i, %if.then.i.i.i.i.i
  %8 = zext i8 %5 to i32
  %arrayidx7.i.i.1.i.i.i = getelementptr [14 x i32], ptr addrspace(3) %1, i32 0, i32 %8
  store i32 0, ptr addrspace(3) %arrayidx7.i.i.1.i.i.i, align 4
  %9 = zext i8 %4 to i32
  %arrayidx12.i = getelementptr [14 x i32], ptr addrspace(3) %2, i32 0, i32 %9
  store i32 0, ptr addrspace(3) %arrayidx12.i, align 4
  store i32 0, ptr addrspace(1) %input.coerce, align 4
  ret void

if.then.i:                                        ; preds = %entry
  store i32 0, ptr addrspace(3) null, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %entry
  %cmp.not.i.i.i.i.not.i = icmp eq i8 %6, 0
  br i1 %cmp.not.i.i.i.i.not.i, label %func.exit.i.i.i, label %if.then.i.i.i.i.i
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
