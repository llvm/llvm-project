; RUN: llc --filetype=obj -O0 -o - %s | llvm-dwarfdump --verify -

; Check that abstract DIE for a subprogram referenced from another compile unit
; is emitted in the correct CU.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define void @a() !dbg !10 {
  br label %for.b.c.c, !dbg !13
  for.b.c.c:
    br label %for.b.c.c
}

!llvm.dbg.cu = !{!0, !6}
!llvm.module.flags = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_20, file: !1, emissionKind: FullDebug, globals: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = !DIGlobalVariable(type: !5)
!5 = !DICompositeType(tag: DW_TAG_class_type)
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_20, file: !7, emissionKind: FullDebug)
!7 = !DIFile(filename: "bar.cpp", directory: "")
!8 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(type: !11, unit: !6)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(scope: !14, inlinedAt: !15)
!14 = distinct !DISubprogram(unit: !6)
!15 = !DILocation(scope: !16, inlinedAt: !25)
!16 = distinct !DISubprogram(type: !11, unit: !6, declaration: !17)
!17 = !DISubprogram(scope: !5, type: !11, spFlags: DISPFlagOptimized, templateParams: !18)
!18 = !{!19}
!19 = !DITemplateTypeParameter(type: !20)
!20 = !DICompositeType(tag: DW_TAG_class_type, scope: !21)
!21 = distinct !DISubprogram(unit: !6, retainedNodes: !22)
!22 = !{!23}
!23 = !DILocalVariable(scope: !21, type: !24)
!24 = !DIBasicType()
!25 = !DILocation(scope: !21, inlinedAt: !26)
!26 = !DILocation(scope: !10)
