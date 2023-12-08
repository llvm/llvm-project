; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-interleave=1 -mattr=+sve -mtriple aarch64-unknown-linux-gnu \
; RUN:     -prefer-predicate-over-epilogue=scalar-epilogue -pass-remarks-missed=loop-vectorize < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=CHECK-REMARKS
; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-interleave=1 -force-target-instruction-cost=1 -mattr=+sve -mtriple aarch64-unknown-linux-gnu \
; RUN:     -prefer-predicate-over-epilogue=scalar-epilogue -pass-remarks-missed=loop-vectorize < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=CHECK-REMARKS

define void @vec_load(i64 %N, ptr nocapture %a, ptr nocapture readonly %b) {
; CHECK-LABEL: @vec_load
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x double>, ptr
; CHECK: call <vscale x 2 x double> @foo_vec(<vscale x 2 x double> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @foo(double %0) #0
  %add = fadd double %1, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, ptr %a, i64 %iv
  store double %add, ptr %arrayidx2, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:                                 ; preds = %for.body, %entry
  ret void
}

define void @vec_scalar(i64 %N, ptr nocapture %a) {
; CHECK-LABEL: @vec_scalar
; CHECK: vector.body:
; CHECK: call <vscale x 2 x double> @foo_vec(<vscale x 2 x double> shufflevector (<vscale x 2 x double> insertelement (<vscale x 2 x double> poison, double 1.000000e+01, i64 0), <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer))
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %0 = call double @foo(double 10.0) #0
  %sub = fsub double %0, 1.000000e+00
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %iv
  store double %sub, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:                                 ; preds = %for.body, %entry
  ret void
}

define void @vec_ptr(i64 %N, ptr noalias %a, ptr readnone %b) {
; CHECK-LABEL: @vec_ptr
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x ptr>, ptr
; CHECK: call <vscale x 2 x i64> @bar_vec(<vscale x 2 x ptr> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep = getelementptr ptr, ptr %b, i64 %iv
  %load = load ptr, ptr %gep
  %call = call i64 @bar(ptr %load) #1
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %call, ptr %arrayidx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

define void @vec_intrinsic(i64 %N, ptr nocapture readonly %a) {
; CHECK-LABEL: @vec_intrinsic
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x double>, ptr
; CHECK: call fast <vscale x 2 x double> @sin_vec_nxv2f64(<vscale x 2 x double> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call fast double @llvm.sin.f64(double %0) #2
  %add = fadd fast double %1, 1.000000e+00
  store double %add, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

; CHECK-REMARKS: UserVF ignored because of invalid costs.
; CHECK-REMARKS-NEXT: t.c:3:10: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): load
; CHECK-REMARKS-NEXT: t.c:3:20: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2): call to llvm.sin.f32
; CHECK-REMARKS-NEXT: t.c:3:30: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): store
define void @vec_sin_no_mapping(ptr noalias nocapture %dst, ptr noalias nocapture readonly %src, i64 %n) {
; CHECK: @vec_sin_no_mapping
; CHECK: call fast <2 x float> @llvm.sin.v2f32
; CHECK-NOT: <vscale x
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %src, i64 %i.07
  %0 = load float, ptr %arrayidx, align 4, !dbg !11
  %1 = tail call fast float @llvm.sin.f32(float %0), !dbg !12
  %arrayidx1 = getelementptr inbounds float, ptr %dst, i64 %i.07
  store float %1, ptr %arrayidx1, align 4, !dbg !13
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

; CHECK-REMARKS: UserVF ignored because of invalid costs.
; CHECK-REMARKS-NEXT: t.c:3:10: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): load
; CHECK-REMARKS-NEXT: t.c:3:30: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2): call to llvm.sin.f32
; CHECK-REMARKS-NEXT: t.c:3:20: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2): call to llvm.sin.f32
; CHECK-REMARKS-NEXT: t.c:3:40: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): store
define void @vec_sin_no_mapping_ite(ptr noalias nocapture %dst, ptr noalias nocapture readonly %src, i64 %n) {
; CHECK: @vec_sin_no_mapping_ite
; CHECK-NOT: <vscale x
; CHECK: ret
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %i.07 = phi i64 [ %inc, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %src, i64 %i.07
  %0 = load float, ptr %arrayidx, align 4, !dbg !11
  %cmp = fcmp ugt float %0, 0.0000
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %1 = tail call fast float @llvm.sin.f32(float %0), !dbg !12
  br label %if.end
if.else:
  %2 = tail call fast float @llvm.sin.f32(float 0.0), !dbg !13
  br label %if.end
if.end:
  %3 = phi float [%1, %if.then], [%2, %if.else]
  %arrayidx1 = getelementptr inbounds float, ptr %dst, i64 %i.07
  store float %3, ptr %arrayidx1, align 4, !dbg !14
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

; CHECK-REMARKS: UserVF ignored because of invalid costs.
; CHECK-REMARKS-NEXT: t.c:3:10: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): load
; CHECK-REMARKS-NEXT: t.c:3:20: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2): call to llvm.sin.f32
; CHECK-REMARKS-NEXT: t.c:3:30: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): store
define void @vec_sin_fixed_mapping(ptr noalias nocapture %dst, ptr noalias nocapture readonly %src, i64 %n) {
; CHECK: @vec_sin_fixed_mapping
; CHECK: call fast <2 x float> @llvm.sin.v2f32
; CHECK-NOT: <vscale x
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %src, i64 %i.07
  %0 = load float, ptr %arrayidx, align 4, !dbg !11
  %1 = tail call fast float @llvm.sin.f32(float %0) #3, !dbg !12
  %arrayidx1 = getelementptr inbounds float, ptr %dst, i64 %i.07
  store float %1, ptr %arrayidx1, align 4, !dbg !13
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

; Even though there are no function mappings attached to the call
; in the loop below we can still vectorize the loop because SVE has
; hardware support in the form of the 'fqsrt' instruction.
define void @vec_sqrt_no_mapping(ptr noalias nocapture %dst, ptr noalias nocapture readonly %src, i64 %n) #0 {
; CHECK: @vec_sqrt_no_mapping
; CHECK: call fast <vscale x 2 x float> @llvm.sqrt.nxv2f32
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %src, i64 %i.07
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call fast float @llvm.sqrt.f32(float %0)
  %arrayidx1 = getelementptr inbounds float, ptr %dst, i64 %i.07
  store float %1, ptr %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}


declare double @foo(double)
declare i64 @bar(ptr)
declare double @llvm.sin.f64(double)
declare float @llvm.sin.f32(float)
declare float @llvm.sqrt.f32(float)

declare <vscale x 2 x double> @foo_vec(<vscale x 2 x double>)
declare <vscale x 2 x i64> @bar_vec(<vscale x 2 x ptr>)
declare <vscale x 2 x double> @sin_vec_nxv2f64(<vscale x 2 x double>)
declare <2 x double> @sin_vec_v2f64(<2 x double>)

attributes #0 = { "vector-function-abi-variant"="_ZGVsNxv_foo(foo_vec)" }
attributes #1 = { "vector-function-abi-variant"="_ZGVsNxv_bar(bar_vec)" }
attributes #2 = { "vector-function-abi-variant"="_ZGVsNxv_llvm.sin.f64(sin_vec_nxv2f64)" }
attributes #3 = { "vector-function-abi-variant"="_ZGV_LLVM_N2v_llvm.sin.f64(sin_vec_v2f64)" }

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}

!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !6, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "t.c", directory: "somedir")
!6 = !{}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "foo", scope: !5, file: !5, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !6)
!10 = !DISubroutineType(types: !6)
!11 = !DILocation(line: 3, column: 10, scope: !9)
!12 = !DILocation(line: 3, column: 20, scope: !9)
!13 = !DILocation(line: 3, column: 30, scope: !9)
!14 = !DILocation(line: 3, column: 40, scope: !9)
