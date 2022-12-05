; RUN: opt -force-vector-interleave=2 -force-vector-width=4 -passes=loop-vectorize -pass-remarks-analysis=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

; CHECK: remark: mixed-precision.c:3:26: floating point conversion changes vector width. Mixed floating point precision requires an up/down cast that will negatively impact performance.
define void @f(float* noalias nocapture %X, i64 %N) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %X, i64 %i
  %0 = load float, float* %arrayidx, align 4
  %conv = fpext float %0 to double, !dbg !9
  %mul = fmul double %conv, 0x3FD5555555555555
  %conv3 = fptrunc double %mul to float
  store float %conv3, float* %arrayidx, align 4
  %inc = add nuw i64 %i, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK: remark: mixed-precision.c:8:8: floating point conversion changes vector width. Mixed floating point precision requires an up/down cast that will negatively impact performance.
; CHECK: remark: mixed-precision.c:7:16: floating point conversion changes vector width. Mixed floating point precision requires an up/down cast that will negatively impact performance.
; CHECK-NOT: remark: mixed-precision.c:7:16: floating point conversion changes vector width. Mixed floating point precision requires an up/down cast that will negatively impact performance.
define void @g(float* noalias nocapture %X, float* noalias nocapture %Y, i64 %N) {
entry:
  %pi = alloca double
  store double 0x400921FB54442D18, double* %pi
  %fac = load double, double* %pi
  br label %for.body

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %X, i64 %i
  %0 = load float, float* %arrayidx, align 4
  %conv = fpext float %0 to double, !dbg !10
  %mul = fmul double %conv, %fac
  %conv1 = fptrunc double %mul to float
  store float %conv1, float* %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds float, float* %Y, i64 %i
  %1 = load float, float* %arrayidx5, align 4
  %conv2 = fpext float %1 to double, !dbg !11
  %mul2 = fmul double %conv2, %fac
  %conv3 = fptrunc double %mul2 to float
  store float %conv3, float* %arrayidx5, align 4
  %inc = add nuw nsw i64 %i, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "mixed-precision.c", directory: "/tmp/mixed-precision.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 3, column: 26, scope: !6)
!10 = !DILocation(line: 7, column: 16, scope: !7)
!11 = !DILocation(line: 8, column: 8, scope: !7)
