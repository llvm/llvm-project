; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Make sure we vectorize this loop:
; int foo(float *a, float *b, int n) {
;   for (int i=0; i<n; ++i)
;     a[i] = b[i] * 3;
; }

;CHECK-LABEL: define i32 @foo
;CHECK: for.body.preheader:
;CHECK: br i1 %cmp.zero, label %scalar.ph, label %vector.memcheck, !dbg [[BODY_LOC:![0-9]+]]
;CHECK: vector.memcheck:
;CHECK: br i1 %memcheck.conflict, label %scalar.ph, label %vector.ph, !dbg [[BODY_LOC]]
;CHECK: load <4 x float>
define i32 @foo(float* nocapture %a, float* nocapture %b, i32 %n) nounwind uwtable ssp {
entry:
  %cmp6 = icmp sgt i32 %n, 0, !dbg !6
  br i1 %cmp6, label %for.body, label %for.end, !dbg !6

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ], !dbg !7
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv, !dbg !7
  %0 = load float, float* %arrayidx, align 4, !dbg !7
  %mul = fmul float %0, 3.000000e+00, !dbg !7
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv, !dbg !7
  store float %mul, float* %arrayidx2, align 4, !dbg !7
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !7
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !7
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !7
  br i1 %exitcond, label %for.end, label %for.body, !dbg !7

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef, !dbg !8
}

; Make sure that we try to vectorize loops with a runtime check if the
; dependency check fails.

; CHECK-LABEL: test_runtime_check
; CHECK:      <4 x float>
define void @test_runtime_check(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float, float* %a, i64 %ind.sum
  %l1 = load float, float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float, float* %a, i64 %ind.sum2
  %l2 = load float, float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}

; CHECK: [[BODY_LOC]] = !DILocation(line: 101, column: 1, scope: !{{.*}})

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!9}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = !DIFile(filename: "test.cpp", directory: "/tmp")
!5 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 99, type: !3, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !9, variables: !2)
!6 = !DILocation(line: 100, column: 1, scope: !5)
!7 = !DILocation(line: 101, column: 1, scope: !5)
!8 = !DILocation(line: 102, column: 1, scope: !5)
!9 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !10,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!10 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!11 = !{i32 2, !"Debug Info Version", i32 3}
