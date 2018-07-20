; RUN: opt < %s -data-layout="e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-n8:16:32" -basicaa -newgvn -S -die | FileCheck %s
; RUN: opt < %s -data-layout="E-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-n32"      -basicaa -newgvn -S -die | FileCheck %s
; memset -> i16 forwarding.
define signext i16 @memset_to_i16_local(i16* %A) nounwind ssp {
entry:
  %conv = bitcast i16* %A to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %conv, i8 1, i64 200, i1 false)
  %arrayidx = getelementptr inbounds i16, i16* %A, i64 42
  %tmp2 = load i16, i16* %arrayidx
  ret i16 %tmp2
; CHECK-LABEL: @memset_to_i16_local(
; CHECK-NOT: load
; CHECK: ret i16 257
}

@GCst = constant {i32, float, i32 } { i32 42, float 14., i32 97 }
@GCst_as1 = addrspace(1) constant {i32, float, i32 } { i32 42, float 14., i32 97 }

; memset -> float forwarding.
define float @memcpy_to_float_local(float* %A) nounwind ssp {
entry:
  %conv = bitcast float* %A to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %conv, i8* bitcast ({i32, float, i32 }* @GCst to i8*), i64 12, i1 false)
  %arrayidx = getelementptr inbounds float, float* %A, i64 1 ; <float*> [#uses=1]
  %tmp2 = load float, float* %arrayidx                   ; <float> [#uses=1]
  ret float %tmp2
; CHECK-LABEL: @memcpy_to_float_local(
; CHECK-NOT: load
; CHECK: ret float 1.400000e+01
}
; memcpy from address space 1
define float @memcpy_to_float_local_as1(float* %A) nounwind ssp {
entry:
  %conv = bitcast float* %A to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.memcpy.p0i8.p1i8.i64(i8* %conv, i8 addrspace(1)* bitcast ({i32, float, i32 } addrspace(1)* @GCst_as1 to i8 addrspace(1)*), i64 12, i1 false)
  %arrayidx = getelementptr inbounds float, float* %A, i64 1 ; <float*> [#uses=1]
  %tmp2 = load float, float* %arrayidx                   ; <float> [#uses=1]
  ret float %tmp2
; CHECK-LABEL: @memcpy_to_float_local_as1(
; CHECK-NOT: load
; CHECK: ret float 1.400000e+01
}

; PR6642
define i32 @memset_to_load() nounwind readnone {
entry:
  %x = alloca [256 x i32], align 4                ; <[256 x i32]*> [#uses=2]
  %tmp = bitcast [256 x i32]* %x to i8*           ; <i8*> [#uses=1]
  call void @llvm.memset.p0i8.i64(i8* align 4 %tmp, i8 0, i64 1024, i1 false)
  %arraydecay = getelementptr inbounds [256 x i32], [256 x i32]* %x, i32 0, i32 0 ; <i32*>
  %tmp1 = load i32, i32* %arraydecay                   ; <i32> [#uses=1]
  ret i32 %tmp1
; CHECK-LABEL: @memset_to_load(
; CHECK: ret i32 0
}
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
declare void @llvm.memcpy.p0i8.p1i8.i64(i8* nocapture, i8 addrspace(1)* nocapture, i64, i1) nounwind
