; RUN: opt -S -scalarrepl < %s | FileCheck %s
; PR7437 - Make sure SROA preserves address space of memcpy when
; hacking on it.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

%struct.anon = type { [1 x float] }

; CHECK-LABEL: define void @Test(
; CHECK: load float, float addrspace(2)*
; CHECK-NEXT: fsub float
; CHECK: store float {{.*}}, float addrspace(2)* 
define void @Test(%struct.anon addrspace(2)* %pPtr) nounwind {
entry:
  %s = alloca %struct.anon, align 4               ; <%struct.anon*> [#uses=3]
  %arrayidx = getelementptr inbounds %struct.anon, %struct.anon addrspace(2)* %pPtr, i64 0 ; <%struct.anon addrspace(2)*> [#uses=1]
  %tmp1 = bitcast %struct.anon* %s to i8*         ; <i8*> [#uses=1]
  %tmp2 = bitcast %struct.anon addrspace(2)* %arrayidx to i8 addrspace(2)* ; <i8 addrspace(2)*> [#uses=1]
  call void @llvm.memcpy.p0i8.p2i8.i64(i8* %tmp1, i8 addrspace(2)* %tmp2, i64 4, i32 4, i1 false)
  %tmp3 = getelementptr inbounds %struct.anon, %struct.anon* %s, i32 0, i32 0 ; <[1 x float]*> [#uses=1]
  %arrayidx4 = getelementptr inbounds [1 x float], [1 x float]* %tmp3, i32 0, i64 0 ; <float*> [#uses=2]
  %tmp5 = load float, float* %arrayidx4                  ; <float> [#uses=1]
  %sub = fsub float %tmp5, 5.000000e+00           ; <float> [#uses=1]
  store float %sub, float* %arrayidx4
  %arrayidx7 = getelementptr inbounds %struct.anon, %struct.anon addrspace(2)* %pPtr, i64 0 ; <%struct.anon addrspace(2)*> [#uses=1]
  %tmp8 = bitcast %struct.anon addrspace(2)* %arrayidx7 to i8 addrspace(2)* ; <i8 addrspace(2)*> [#uses=1]
  %tmp9 = bitcast %struct.anon* %s to i8*         ; <i8*> [#uses=1]
  call void @llvm.memcpy.p2i8.p0i8.i64(i8 addrspace(2)* %tmp8, i8* %tmp9, i64 4, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p2i8.i64(i8* nocapture, i8 addrspace(2)* nocapture, i64, i32, i1) nounwind

declare void @llvm.memcpy.p2i8.p0i8.i64(i8 addrspace(2)* nocapture, i8* nocapture, i64, i32, i1) nounwind

