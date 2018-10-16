; RUN: opt -S -lcssa < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Reproducer for PR39019.
;
; Verify that the llvm.dbg.value in the %for.cond.cleanup2 block is rewritten
; to use the PHI node for %add that is created by LCSSA.

; CHECK-LABEL: for.cond.cleanup2:
; CHECK-NEXT: [[PN:%[^ ]*]] = phi i32 [ %add.lcssa, %for.cond.cleanup1 ]
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PN]], metadata [[VAR:![0-9]+]], metadata !DIExpression())
; CHECK-NEXT: call void @bar(i32 [[PN]])

; CHECK-LABEL: for.body:
; CHECK: %add = add nsw i32 0, 2
; CHECK: call void @llvm.dbg.value(metadata i32 %add, metadata [[VAR]], metadata !DIExpression())

; CHECK: [[VAR]] = !DILocalVariable(name: "sum",

; Function Attrs: nounwind
define void @foo() #0 !dbg !6 {
entry:
  br label %for.cond.preheader, !dbg !12

for.cond.preheader:                               ; preds = %for.cond.cleanup1, %entry
  br label %for.body, !dbg !12

for.cond.cleanup2:                                ; preds = %for.cond.cleanup1
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !12
  tail call void @bar(i32 %add) #0, !dbg !12
  ret void, !dbg !12

for.cond.cleanup1:                                ; preds = %for.body
  br i1 false, label %for.cond.preheader, label %for.cond.cleanup2, !dbg !12

for.body:                                         ; preds = %for.body, %for.cond.preheader
  %add = add nsw i32 0, 2, !dbg !12
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !12
  br i1 false, label %for.body, label %for.cond.cleanup1, !dbg !12
}

; Function Attrs: nounwind
declare void @bar(i32) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 8.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !7, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "sum", scope: !10, file: !1, line: 11, type: !11)
!10 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 0)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 0, scope: !10)
