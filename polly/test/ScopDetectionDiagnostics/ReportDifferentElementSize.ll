; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; 1 void differenttypes(char *A)
; 2 {
; 3   for (long i = 0; i < 1024; ++i)
; 4     ((float*)A)[i] = ((double*)A)[i];
; 5 }

; CHECK: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: /tmp/test.c:4:14: The array "A" is accessed through elements that differ in size
; CHECK-NEXT: remark: /tmp/test.c:4:32: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @differenttypes(i8* nocapture %A)  !dbg !4 {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i64 [ 0, %entry ], [ %tmp11, %for.body ]
  %tmp = shl i64 %i.05, 3, !dbg !15
  %uglygep = getelementptr i8, i8* %A, i64 %tmp
  %arrayidx = bitcast i8* %uglygep to double*, !dbg !16
  %tmp9 = shl i64 %i.05, 2, !dbg !15
  %uglygep7 = getelementptr i8, i8* %A, i64 %tmp9
  %arrayidx1 = bitcast i8* %uglygep7 to float*, !dbg !17
  %tmp10 = load double, double* %arrayidx, align 8, !dbg !16, !tbaa !18
  %conv = fptrunc double %tmp10 to float, !dbg !16
  store float %conv, float* %arrayidx1, align 4, !dbg !17, !tbaa !22
  %tmp11 = add nsw i64 %i.05, 1, !dbg !24
  %exitcond = icmp eq i64 %tmp11, 1024, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.body
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: 2, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "differenttypes", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "/tmp/test.c", directory: "/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.6.0 "}
!10 = !DILocation(line: 3, column: 20, scope: !11)
!11 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !12)
!12 = !DILexicalBlockFile(discriminator: 1, file: !1, scope: !13)
!13 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !14)
!14 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !4)
!15 = !DILocation(line: 4, column: 32, scope: !13)
!16 = !DILocation(line: 4, column: 22, scope: !13)
!17 = !DILocation(line: 4, column: 14, scope: !13)
!18 = !{!19, !19, i64 0}
!19 = !{!"double", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !20, i64 0}
!24 = !DILocation(line: 3, column: 30, scope: !13)
!25 = !DILocation(line: 5, column: 1, scope: !4)
