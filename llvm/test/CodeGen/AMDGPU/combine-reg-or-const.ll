; RUN: llc  -mtriple=amdgcn-amd-amdhsa -o - %s | FileCheck %s

; The OR instruction should not be eliminated by the "OR Combine" DAG optimization.

; CHECK-LABEL: _Z11test_kernelPii:
; CHECK: s_mul_i32
; CHECK: s_sub_i32
; CHECK: s_and_b32 [[S1:s[0-9]+]], {{s[0-9]+}}, 0xffff
; CHECK: s_add_i32 [[S2:s[0-9]+]], {{s[0-9]+}}, [[S1]]
; CHECK: s_or_b32 {{s[0-9]+}}, [[S2]], 0xc0

define protected amdgpu_kernel void @_Z11test_kernelPii(ptr addrspace(1) nocapture %Ad.coerce, i32 %s) local_unnamed_addr #5 {
entry:
  %cmp = icmp eq i32 %s, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %rem.lhs.trunc = trunc i32 %s to i16
  %rem4 = urem i16 %rem.lhs.trunc, 12
  %rem.zext = zext i16 %rem4 to i32
  %idxprom = zext i32 %s to i64
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %Ad.coerce, i64 %idxprom
  %div = lshr i32 %rem.zext, 3
  %or = or i32 %rem.zext, 192
  %add = add nuw nsw i32 %or, %div
  store i32 %add, ptr addrspace(1) %arrayidx3, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
