; RUN: opt %loadPolly \
; RUN: -polly-analyze-read-only-scalars=false -polly-codegen -S < %s | \
; RUN: FileCheck %s

; RUN: opt %loadPolly \
; RUN: -polly-analyze-read-only-scalars=true -polly-codegen -S < %s | \
; RUN: FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(float* %A, i64 %N) #0 !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata float* %A, metadata !14, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata i64 %N, metadata !16, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !20
  %cmp1 = icmp sgt i64 %N, 0, !dbg !20
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !20

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body, !dbg !20

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %0 = phi i64 [ 0, %for.body.lr.ph ], [ %1, %for.body ], !dbg !21
  %arrayidx = getelementptr float, float* %A, i64 %0, !dbg !21
  %conv = sitofp i64 %0 to float, !dbg !21
  store float %conv, float* %arrayidx, align 4, !dbg !21
  %1 = add nsw i64 %0, 1, !dbg !20
  tail call void @llvm.dbg.value(metadata i64 %1, metadata !18, metadata !DIExpression()), !dbg !20
  %exitcond = icmp ne i64 %1, %N, !dbg !20
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !20

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !20

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void, !dbg !22
}

; CHECK: polly.split_new_and_old:

; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK: tail call void @llvm.dbg.value
; CHECK-NOT: tail call void @llvm.dbg.value

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "loop.c", directory: "/home/grosser/Projects/polly/git/tools/polly")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "loop.c", directory: "/home/grosser/Projects/polly/git/tools/polly")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !10}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.5 "}
!14 = !DILocalVariable(name: "A", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!15 = !DILocation(line: 1, scope: !4)
!16 = !DILocalVariable(name: "N", line: 1, arg: 2, scope: !4, file: !5, type: !10)
!17 = !{i64 0}
!18 = !DILocalVariable(name: "i", line: 2, scope: !19, file: !5, type: !10)
!19 = distinct !DILexicalBlock(line: 2, column: 0, file: !1, scope: !4)
!20 = !DILocation(line: 2, scope: !19)
!21 = !DILocation(line: 3, scope: !19)
!22 = !DILocation(line: 4, scope: !4)
