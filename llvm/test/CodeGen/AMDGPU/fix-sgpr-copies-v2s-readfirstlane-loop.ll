; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; Test that the SIFixSGPRCopies cost model keeps V_READFIRSTLANE_B32 outside
; the loop when it feeds an unconvertible SALU instruction (SI_INIT_M0) inside
; the loop.
;
; Without the cost model accounting for unavoidable readfirstlanes and loop
; depth penalty, the VGPR-to-SGPR copy would be converted to VALU, causing
; legalizeOperands to insert V_READFIRSTLANE_B32 at the SI_INIT_M0 use site
; inside the loop.

; The v_readfirstlane_b32 must appear before the loop, not inside it.
; CHECK-LABEL: readfirstlane_in_loop:
; CHECK:       ; %bb.0:
; CHECK:       v_readfirstlane_b32
; CHECK:       ; %loop
; CHECK-NOT:   v_readfirstlane_b32
; CHECK:       s_mov_b32 m0,
; CHECK-NOT:   v_readfirstlane_b32
; CHECK:       s_endpgm

@a = internal unnamed_addr addrspace(3) global [2048 x i8] poison, align 16

declare void @llvm.amdgcn.load.to.lds.p1(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)

define amdgpu_kernel void @readfirstlane_in_loop(ptr addrspace(1) noalias noundef nocapture readonly %inp, ptr addrspace(1) noalias noundef nocapture writeonly %out) #0 !reqd_work_group_size !0 {
entry:
  %tid = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
  %wgid = call i32 @llvm.amdgcn.workgroup.id.x()
  %wgid.zext = zext nneg i32 %wgid to i64
  %wgoff = shl nsw i64 %wgid.zext, 11
  %tidoff = shl nsw i32 %tid, 4
  %tidoff.zext = zext nneg i32 %tid to i64
  %globaloff = add nsw nuw i64 %wgoff, %tidoff.zext
  %src.base = getelementptr inbounds nuw i8, ptr addrspace(1) %inp, i64 %globaloff
  %dst.base = getelementptr inbounds nuw i8, ptr addrspace(1) %out, i64 %globaloff
  %waveid = lshr i32 %tid, 6
  %waveoff = shl nsw nuw i32 %waveid, 4
  %our.lds = getelementptr inbounds nuw i8, ptr addrspace(3) @a, i32 %waveoff
  %my.lds = getelementptr inbounds nuw i8, ptr addrspace(3) @a, i32 %tidoff
  br label %loop

loop:
  %k = phi i32 [%k.next, %loop], [0, %entry]
  %src = phi ptr addrspace(1) [%src.next, %loop], [%src.base, %entry]
  %dst = phi ptr addrspace(1) [%dst.next, %loop], [%dst.base, %entry]
  call void @llvm.amdgcn.load.to.lds.p1(ptr addrspace(1) %src, ptr addrspace(3) %our.lds, i32 16, i32 0, i32 0)
  %v = load <4 x i32>, ptr addrspace(3) %my.lds, align 16
  store <4 x i32> %v, ptr addrspace(1) %dst, align 16
  %k.next = add nsw nuw i32 %k, 1
  %src.next = getelementptr inbounds nuw i8, ptr addrspace(1) %src, i64 65536
  %dst.next = getelementptr inbounds nuw i8, ptr addrspace(1) %dst, i64 65536
  %cond = icmp samesign ult i32 %k, 9
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="128,128" }
!0 = !{i32 128, i32 1, i32 1}
