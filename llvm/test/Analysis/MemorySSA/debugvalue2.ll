; RUN: opt -passes='print<memoryssa>' -disable-output %s 2>&1 | FileCheck %s

; Note that the test crashes the MemorySSA verification when doing loop-rotate,
; if debuginfo is modelled in MemorySSA, due to the fact that MemorySSA is not
; updated when adding/removing debuginfo intrinsics.

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @overflow_iter_var
; CHECK-NOT: MemoryDef
define void @overflow_iter_var(i1 %arg) !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata i16 0, metadata !16, metadata !DIExpression()), !dbg !18
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  call void @llvm.dbg.value(metadata i16 0, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i16 undef, metadata !20, metadata !DIExpression()), !dbg !21
  br i1 %arg, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  %0 = load i16, ptr undef, align 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, nameTableKind: None)
!1 = !DIFile(filename: "2_loops.c", directory: "/")
!2 = !{}
!3 = !{}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 4096, elements: !8)
!7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 256)
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = distinct !DISubprogram(name: "overflow_iter_var", scope: !1, file: !1, line: 20, type: !12, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14}
!14 = !DIBasicType(name: "unsigned int", size: 16, encoding: DW_ATE_unsigned)
!16 = !DILocalVariable(name: "i", scope: !17, file: !1, line: 23, type: !14)
!17 = distinct !DILexicalBlock(scope: !11, file: !1, line: 23, column: 3)
!18 = !DILocation(line: 0, scope: !17)
!20 = !DILocalVariable(name: "stop1", arg: 1, scope: !11, file: !1, line: 20, type: !14)
!21 = !DILocation(line: 0, scope: !11)
