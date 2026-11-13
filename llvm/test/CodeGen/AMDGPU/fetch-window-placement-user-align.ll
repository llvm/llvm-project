; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck --check-prefixes=CHECK,ON %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-enable-fetch-window-placement=0 < %s | FileCheck --check-prefixes=CHECK,OFF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-fetch-window-print -o /dev/null < %s 2>&1 | FileCheck --check-prefix=PRINT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -align-loops=64 -amdgpu-fetch-window-print -o /dev/null < %s 2>&1 | FileCheck --allow-empty --check-prefix=ALIGN-LOOPS %s

; Loops with llvm.loop.align are not moved by the fetch window placement. The
; one s_nop padding limit of the fetch window alignment does not apply to a
; larger user alignment, and the placement lifts it for an equal one.

; PRINT-NOT: plain{{.*}}user-aligned
; PRINT: align64 loop %bb.{{[0-9]+}} user-aligned, skip
; PRINT: align32 loop %bb.{{[0-9]+}} user-aligned, skip

; ALIGN-LOOPS-NOT: fetch-window:

; CHECK-LABEL: plain:
; CHECK: .p2align 5, , 4
; CHECK-NEXT: .LBB0_2: ; %for.body
; CHECK-LABEL: align64:
; CHECK: .p2align 6{{$}}
; CHECK-NEXT: .LBB1_2: ; %for.body
; CHECK-LABEL: align32:
; ON: .p2align 5{{$}}
; OFF: .p2align 5, , 4
; CHECK-NEXT: .LBB2_2: ; %for.body

define amdgpu_kernel void @plain(ptr addrspace(1) %In.coerce, ptr addrspace(1) %Out.coerce, i32 %N) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.x()
  %idx.ext = zext nneg i32 %0 to i64
  %add.ptr = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %In.coerce, i64 %idx.ext
  %cmp.not42 = icmp eq i32 %N, 0
  br i1 %cmp.not42, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %Acc.sroa.0.0.lcssa = phi <2 x float> [ zeroinitializer, %entry ], [ %add.i32, %for.body ]
  %shift = shufflevector <2 x float> %Acc.sroa.0.0.lcssa, <2 x float> poison, <2 x i32> <i32 1, i32 poison>
  %foldExtExtBinop = fadd contract <2 x float> %Acc.sroa.0.0.lcssa, %shift
  %add = extractelement <2 x float> %foldExtExtBinop, i64 0
  %arrayidx17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %Out.coerce, i64 %idx.ext
  store float %add, ptr addrspace(1) %arrayidx17, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %I.044 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %Acc.sroa.0.043 = phi <2 x float> [ %add.i32, %for.body ], [ zeroinitializer, %entry ]
  %mul = shl i32 %I.044, 6
  %idx.ext3 = zext i32 %mul to i64
  %add.ptr4 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %add.ptr, i64 %idx.ext3
  %1 = load float, ptr addrspace(1) %add.ptr4, align 4
  %arrayidx5 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 1024
  %2 = load float, ptr addrspace(1) %arrayidx5, align 4
  %A.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %1, i64 0
  %A.sroa.0.4.vec.insert = insertelement <2 x float> %A.sroa.0.0.vec.insert, float %2, i64 1
  %arrayidx6 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 2048
  %3 = load float, ptr addrspace(1) %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 3072
  %4 = load float, ptr addrspace(1) %arrayidx7, align 4
  %B.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %3, i64 0
  %B.sroa.0.4.vec.insert = insertelement <2 x float> %B.sroa.0.0.vec.insert, float %4, i64 1
  %arrayidx8 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 4096
  %5 = load float, ptr addrspace(1) %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 5120
  %6 = load float, ptr addrspace(1) %arrayidx9, align 4
  %C.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %5, i64 0
  %C.sroa.0.4.vec.insert = insertelement <2 x float> %C.sroa.0.0.vec.insert, float %6, i64 1
  %arrayidx10 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 6144
  %7 = load float, ptr addrspace(1) %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 7168
  %8 = load float, ptr addrspace(1) %arrayidx11, align 4
  %D.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %7, i64 0
  %D.sroa.0.4.vec.insert = insertelement <2 x float> %D.sroa.0.0.vec.insert, float %8, i64 1
  %add.i = fadd contract <2 x float> %Acc.sroa.0.043, %A.sroa.0.4.vec.insert
  %add.i30 = fadd contract <2 x float> %add.i, %B.sroa.0.4.vec.insert
  %add.i31 = fadd contract <2 x float> %add.i30, %C.sroa.0.4.vec.insert
  %add.i32 = fadd contract <2 x float> %add.i31, %D.sroa.0.4.vec.insert
  %inc = add nuw i32 %I.044, 1
  %cmp.not = icmp eq i32 %inc, %N
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

define amdgpu_kernel void @align64(ptr addrspace(1) %In.coerce, ptr addrspace(1) %Out.coerce, i32 %N) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.x()
  %idx.ext = zext nneg i32 %0 to i64
  %add.ptr = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %In.coerce, i64 %idx.ext
  %cmp.not42 = icmp eq i32 %N, 0
  br i1 %cmp.not42, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %Acc.sroa.0.0.lcssa = phi <2 x float> [ zeroinitializer, %entry ], [ %add.i32, %for.body ]
  %shift = shufflevector <2 x float> %Acc.sroa.0.0.lcssa, <2 x float> poison, <2 x i32> <i32 1, i32 poison>
  %foldExtExtBinop = fadd contract <2 x float> %Acc.sroa.0.0.lcssa, %shift
  %add = extractelement <2 x float> %foldExtExtBinop, i64 0
  %arrayidx17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %Out.coerce, i64 %idx.ext
  store float %add, ptr addrspace(1) %arrayidx17, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %I.044 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %Acc.sroa.0.043 = phi <2 x float> [ %add.i32, %for.body ], [ zeroinitializer, %entry ]
  %mul = shl i32 %I.044, 6
  %idx.ext3 = zext i32 %mul to i64
  %add.ptr4 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %add.ptr, i64 %idx.ext3
  %1 = load float, ptr addrspace(1) %add.ptr4, align 4
  %arrayidx5 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 1024
  %2 = load float, ptr addrspace(1) %arrayidx5, align 4
  %A.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %1, i64 0
  %A.sroa.0.4.vec.insert = insertelement <2 x float> %A.sroa.0.0.vec.insert, float %2, i64 1
  %arrayidx6 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 2048
  %3 = load float, ptr addrspace(1) %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 3072
  %4 = load float, ptr addrspace(1) %arrayidx7, align 4
  %B.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %3, i64 0
  %B.sroa.0.4.vec.insert = insertelement <2 x float> %B.sroa.0.0.vec.insert, float %4, i64 1
  %arrayidx8 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 4096
  %5 = load float, ptr addrspace(1) %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 5120
  %6 = load float, ptr addrspace(1) %arrayidx9, align 4
  %C.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %5, i64 0
  %C.sroa.0.4.vec.insert = insertelement <2 x float> %C.sroa.0.0.vec.insert, float %6, i64 1
  %arrayidx10 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 6144
  %7 = load float, ptr addrspace(1) %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 7168
  %8 = load float, ptr addrspace(1) %arrayidx11, align 4
  %D.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %7, i64 0
  %D.sroa.0.4.vec.insert = insertelement <2 x float> %D.sroa.0.0.vec.insert, float %8, i64 1
  %add.i = fadd contract <2 x float> %Acc.sroa.0.043, %A.sroa.0.4.vec.insert
  %add.i30 = fadd contract <2 x float> %add.i, %B.sroa.0.4.vec.insert
  %add.i31 = fadd contract <2 x float> %add.i30, %C.sroa.0.4.vec.insert
  %add.i32 = fadd contract <2 x float> %add.i31, %D.sroa.0.4.vec.insert
  %inc = add nuw i32 %I.044, 1
  %cmp.not = icmp eq i32 %inc, %N
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body, !llvm.loop !2
}

define amdgpu_kernel void @align32(ptr addrspace(1) %In.coerce, ptr addrspace(1) %Out.coerce, i32 %N) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.x()
  %idx.ext = zext nneg i32 %0 to i64
  %add.ptr = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %In.coerce, i64 %idx.ext
  %cmp.not42 = icmp eq i32 %N, 0
  br i1 %cmp.not42, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %Acc.sroa.0.0.lcssa = phi <2 x float> [ zeroinitializer, %entry ], [ %add.i32, %for.body ]
  %shift = shufflevector <2 x float> %Acc.sroa.0.0.lcssa, <2 x float> poison, <2 x i32> <i32 1, i32 poison>
  %foldExtExtBinop = fadd contract <2 x float> %Acc.sroa.0.0.lcssa, %shift
  %add = extractelement <2 x float> %foldExtExtBinop, i64 0
  %arrayidx17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %Out.coerce, i64 %idx.ext
  store float %add, ptr addrspace(1) %arrayidx17, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %I.044 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %Acc.sroa.0.043 = phi <2 x float> [ %add.i32, %for.body ], [ zeroinitializer, %entry ]
  %mul = shl i32 %I.044, 6
  %idx.ext3 = zext i32 %mul to i64
  %add.ptr4 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %add.ptr, i64 %idx.ext3
  %1 = load float, ptr addrspace(1) %add.ptr4, align 4
  %arrayidx5 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 1024
  %2 = load float, ptr addrspace(1) %arrayidx5, align 4
  %A.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %1, i64 0
  %A.sroa.0.4.vec.insert = insertelement <2 x float> %A.sroa.0.0.vec.insert, float %2, i64 1
  %arrayidx6 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 2048
  %3 = load float, ptr addrspace(1) %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 3072
  %4 = load float, ptr addrspace(1) %arrayidx7, align 4
  %B.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %3, i64 0
  %B.sroa.0.4.vec.insert = insertelement <2 x float> %B.sroa.0.0.vec.insert, float %4, i64 1
  %arrayidx8 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 4096
  %5 = load float, ptr addrspace(1) %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 5120
  %6 = load float, ptr addrspace(1) %arrayidx9, align 4
  %C.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %5, i64 0
  %C.sroa.0.4.vec.insert = insertelement <2 x float> %C.sroa.0.0.vec.insert, float %6, i64 1
  %arrayidx10 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 6144
  %7 = load float, ptr addrspace(1) %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds nuw i8, ptr addrspace(1) %add.ptr4, i64 7168
  %8 = load float, ptr addrspace(1) %arrayidx11, align 4
  %D.sroa.0.0.vec.insert = insertelement <2 x float> poison, float %7, i64 0
  %D.sroa.0.4.vec.insert = insertelement <2 x float> %D.sroa.0.0.vec.insert, float %8, i64 1
  %add.i = fadd contract <2 x float> %Acc.sroa.0.043, %A.sroa.0.4.vec.insert
  %add.i30 = fadd contract <2 x float> %add.i, %B.sroa.0.4.vec.insert
  %add.i31 = fadd contract <2 x float> %add.i30, %C.sroa.0.4.vec.insert
  %add.i32 = fadd contract <2 x float> %add.i31, %D.sroa.0.4.vec.insert
  %inc = add nuw i32 %I.044, 1
  %cmp.not = icmp eq i32 %inc, %N
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body, !llvm.loop !4
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1, !3}
!3 = !{!"llvm.loop.align", i32 64}
!4 = distinct !{!4, !1, !5}
!5 = !{!"llvm.loop.align", i32 32}
