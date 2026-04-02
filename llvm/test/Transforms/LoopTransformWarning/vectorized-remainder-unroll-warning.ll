; RUN: opt < %s -passes='loop-vectorize,transform-warning' -force-vector-width=4 -force-vector-interleave=1 -disable-output -pass-remarks-missed=transform-warning 2>&1 | FileCheck %s
; RUN: opt < %s -passes='loop-vectorize,transform-warning' -force-vector-width=4 -force-vector-interleave=1 -disable-output -pass-remarks-output=%t.yaml 2>&1
; RUN: FileCheck --input-file=%t.yaml %s --check-prefix=YAML
;
; Verify that when a #pragma unroll loop gets vectorized and the scalar
; remainder inherits the unroll metadata, WarnMissedTransforms emits a
; more informative message for the remainder loop instead of the generic
; "loop not unrolled" warning.
;
; The vectorized loop gets a "vectorized loop" warning (since it has
; vector_body but not scalar_remainder), while the remainder loop
; gets the improved "scalar remainder" message.

; CHECK: warning: {{.*}} vectorized loop not unrolled: the optimizer was unable to perform the requested transformation
; CHECK: warning: {{.*}} scalar remainder loop after vectorization not unrolled: the optimizer was unable to perform the requested transformation

; YAML:      --- !Failure
; YAML-NEXT: Pass:            transform-warning
; YAML-NEXT: Name:            FailedRequestedUnrolling
; YAML:      Function:        pragma_unroll_vectorized
; YAML:      Args:
; YAML:        - String:          'vectorized loop not unrolled:
; YAML:      --- !Failure
; YAML-NEXT: Pass:            transform-warning
; YAML-NEXT: Name:            FailedRequestedUnrolling
; YAML:      Function:        pragma_unroll_vectorized
; YAML:      Args:
; YAML:        - String:          'scalar remainder loop after vectorization not unrolled:

define void @pragma_unroll_vectorized(ptr noalias %A, ptr noalias %B, i32 %n) !dbg !4 {
entry:
  %cmp = icmp sgt i32 %n, 0, !dbg !8
  br i1 %cmp, label %for.body.preheader, label %for.end, !dbg !8

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.preheader ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %iv, !dbg !9
  %val = load i32, ptr %arrayidx, align 4, !dbg !9
  %add = add nsw i32 %val, 1, !dbg !9
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %iv, !dbg !9
  store i32 %add, ptr %arrayidx2, align 4, !dbg !9
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %wide.trip.count, !dbg !8
  br i1 %exitcond, label %for.end, label %for.body, !dbg !8, !llvm.loop !10

for.end:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, file: !1)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "pragma_unroll_vectorized", line: 5, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !1, type: !5, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = distinct !DILexicalBlock(line: 7, column: 3, file: !1, scope: !4)
!8 = !DILocation(line: 7, column: 3, scope: !7)
!9 = !DILocation(line: 8, column: 5, scope: !7)
!10 = distinct !{!10, !12}
!11 = !DILocation(line: 10, column: 1, scope: !4)
!12 = !{!"llvm.loop.unroll.enable"}
