; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.vec2 = type { <4 x i32>, <4 x i32> }
%struct.vec2plusi = type { <4 x i32>, <4 x i32>, i32 }

@glob1 = global %struct.vec2 zeroinitializer, align 16
@glob2 = global %struct.vec2plusi zeroinitializer, align 16

define void @write24to28(i32* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @write24to28(
entry:
  %arrayidx0 = getelementptr inbounds i32, i32* %p, i64 1
  %p3 = bitcast i32* %arrayidx0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 24, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 28, i1 false)
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 7
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

define void @write28to32(i32* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @write28to32(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 28, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 4 %p3, i8 0, i64 32, i1 false)
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 7
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

define void @dontwrite28to32memset(i32* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @dontwrite28to32memset(
entry:
  %p3 = bitcast i32* %p to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 16 %p3, i8 0, i64 32, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 16 %p3, i8 0, i64 32, i1 false)
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 7
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

define void @write32to36(%struct.vec2plusi* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @write32to36(
entry:
  %0 = bitcast %struct.vec2plusi* %p to i8*
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2plusi* @glob2 to i8*), i64 32, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2plusi* @glob2 to i8*), i64 36, i1 false)
  %c = getelementptr inbounds %struct.vec2plusi, %struct.vec2plusi* %p, i64 0, i32 2
  store i32 1, i32* %c, align 4
  ret void
}

define void @write16to32(%struct.vec2* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @write16to32(
entry:
  %0 = bitcast %struct.vec2* %p to i8*
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2* @glob1 to i8*), i64 16, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2* @glob1 to i8*), i64 32, i1 false)
  %c = getelementptr inbounds %struct.vec2, %struct.vec2* %p, i64 0, i32 1
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32>* %c, align 4
  ret void
}

define void @dontwrite28to32memcpy(%struct.vec2* nocapture %p) nounwind uwtable ssp {
; CHECK-LABEL: @dontwrite28to32memcpy(
entry:
  %0 = bitcast %struct.vec2* %p to i8*
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2* @glob1 to i8*), i64 32, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast (%struct.vec2* @glob1 to i8*), i64 32, i1 false)
  %arrayidx1 = getelementptr inbounds %struct.vec2, %struct.vec2* %p, i64 0, i32 0, i64 7
  store i32 1, i32* %arrayidx1, align 4
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

%struct.trapframe = type { i64, i64, i64 }

; bugzilla 11455 - make sure negative GEP's don't break this optimisation
; CHECK-LABEL: @cpu_lwp_fork(
define void @cpu_lwp_fork(%struct.trapframe* %md_regs, i64 %pcb_rsp0) nounwind uwtable noinline ssp {
entry:
  %0 = inttoptr i64 %pcb_rsp0 to %struct.trapframe*
  %add.ptr = getelementptr inbounds %struct.trapframe, %struct.trapframe* %0, i64 -1
  %1 = bitcast %struct.trapframe* %add.ptr to i8*
  %2 = bitcast %struct.trapframe* %md_regs to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 24, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 24, i1 false)
  %tf_trapno = getelementptr inbounds %struct.trapframe, %struct.trapframe* %0, i64 -1, i32 1
  store i64 3, i64* %tf_trapno, align 8
  ret void
}

define void @write16To23AndThen24To31(i64* nocapture %P, i64 %n64, i32 %n32, i16 %n16, i8 %n8) {
entry:
; CHECK-LABEL: @write16To23AndThen24To31(
; CHECK: tail call void @llvm.memset.p0i8.i64(i8* align 8 %mybase0, i8 0, i64 16, i1 false)

  %base0 = bitcast i64* %P to i8*
  %mybase0 = getelementptr inbounds i8, i8* %base0, i64 0
  tail call void @llvm.memset.p0i8.i64(i8* align 8 %mybase0, i8 0, i64 32, i1 false)

  %base64_2 = getelementptr inbounds i64, i64* %P, i64 2
  %base64_3 = getelementptr inbounds i64, i64* %P, i64 3

  store i64 3, i64* %base64_2
  store i64 3, i64* %base64_3
  ret void
}
