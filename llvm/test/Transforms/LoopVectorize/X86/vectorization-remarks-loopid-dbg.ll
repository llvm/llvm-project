; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-unknown-linux -S -pass-remarks='loop-vectorize' 2>&1 | FileCheck -check-prefix=VECTORIZED %s
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=4 -mtriple=x86_64-unknown-linux -S -pass-remarks='loop-vectorize' 2>&1 | FileCheck -check-prefix=UNROLLED %s
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=1 -mtriple=x86_64-unknown-linux -S -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck -check-prefix=NONE %s

; RUN: llc < %s -mtriple x86_64-pc-linux-gnu -o - | FileCheck -check-prefix=DEBUG-OUTPUT %s
; DEBUG-OUTPUT-NOT: .loc
; DEBUG-OUTPUT-NOT: {{.*}}.debug_info

; VECTORIZED: remark: vectorization-remarks.c:17:8: vectorized loop (vectorization width: 4, interleaved count: 1)
; UNROLLED: remark: vectorization-remarks.c:17:8: interleaved loop (interleaved count: 4)
; NONE: remark: vectorization-remarks.c:17:8: loop not vectorized: vectorization and interleaving are explicitly disabled, or the loop has already been vectorized

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @foo(i32 %n) #0 !dbg !4 {
entry:
  %diff = alloca i32, align 4
  %cb = alloca [16 x i8], align 16
  %cc = alloca [16 x i8], align 16
  store i32 0, ptr %diff, align 4, !tbaa !11
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %add8 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [16 x i8], ptr %cb, i64 0, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !21
  %conv = sext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds [16 x i8], ptr %cc, i64 0, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1, !tbaa !21
  %conv3 = sext i8 %1 to i32
  %sub = sub i32 %conv, %conv3
  %add = add nsw i32 %sub, %add8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !25

for.end:                                          ; preds = %for.body
  store i32 %add, ptr %diff, align 4, !tbaa !11
  call void @ibar(ptr %diff) #2
  ret i32 0
}

declare void @ibar(ptr) #1

!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}
!llvm.dbg.cu = !{!24}

!1 = !DIFile(filename: "vectorization-remarks.c", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !24, scopeLine: 6, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "vectorization-remarks.c", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5.0 "}
!10 = !DILocation(line: 8, column: 3, scope: !4)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 17, column: 8, scope: !16)
!16 = distinct !DILexicalBlock(line: 17, column: 8, file: !1, scope: !17)
!17 = distinct !DILexicalBlock(line: 17, column: 8, file: !1, scope: !18)
!18 = distinct !DILexicalBlock(line: 17, column: 3, file: !1, scope: !4)
!19 = !DILocation(line: 18, column: 5, scope: !20)
!20 = distinct !DILexicalBlock(line: 17, column: 27, file: !1, scope: !18)
!21 = !{!13, !13, i64 0}
!22 = !DILocation(line: 20, column: 3, scope: !4)
!23 = !DILocation(line: 21, column: 3, scope: !4)
!24 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, emissionKind: NoDebug)
!25 = !{!25, !15}
