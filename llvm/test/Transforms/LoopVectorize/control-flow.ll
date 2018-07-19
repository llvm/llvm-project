; RUN: opt < %s -loop-vectorize -force-vector-width=4 -S -pass-remarks-missed='loop-vectorize' 2>&1 | FileCheck %s

; C/C++ code for control flow test
; int test(int *A, int Length) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] > 10.0) goto end;
;     A[i] = 0;
;   }
; end:
;   return 0;
; }

; CHECK: remark: source.cpp:5:9: loop not vectorized: loop control flow is not understood by vectorizer
; CHECK: remark: source.cpp:5:9: loop not vectorized

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define i32 @_Z4testPii(i32* nocapture %A, i32 %Length) #0 !dbg !4 {
entry:
  %cmp8 = icmp sgt i32 %Length, 0, !dbg !10
  br i1 %cmp8, label %for.body.preheader, label %end, !dbg !10

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !12

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv, !dbg !12
  %0 = load i32, i32* %arrayidx, align 4, !dbg !12, !tbaa !15
  %cmp1 = icmp sgt i32 %0, 10, !dbg !12
  br i1 %cmp1, label %end.loopexit, label %if.else, !dbg !12

if.else:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !dbg !19, !tbaa !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %1 = trunc i64 %indvars.iv.next to i32, !dbg !10
  %cmp = icmp slt i32 %1, %Length, !dbg !10
  br i1 %cmp, label %for.body, label %end.loopexit, !dbg !10

end.loopexit:                                     ; preds = %if.else, %for.body
  br label %end

end:                                              ; preds = %end.loopexit, %entry
  ret i32 0, !dbg !20
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0", isOptimized: true, runtimeVersion: 6, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "source.cpp", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "source.cpp", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5.0"}
!10 = !DILocation(line: 3, column: 8, scope: !11)
!11 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !4)
!12 = !DILocation(line: 5, column: 9, scope: !13)
!13 = distinct !DILexicalBlock(line: 5, column: 9, file: !1, scope: !14)
!14 = distinct !DILexicalBlock(line: 4, column: 3, file: !1, scope: !11)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 8, column: 7, scope: !13)
!20 = !DILocation(line: 12, column: 3, scope: !4)
