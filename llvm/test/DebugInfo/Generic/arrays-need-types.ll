; RUN: opt %s -o - -S --passes=verify 2>&1 | FileCheck %s

; CHECK:      array types must have a base type
; CHECK-NEXT: !DICompositeType(tag: DW_TAG_array_type,
; CHECK-NEXT: warning: ignoring invalid debug info

declare void @llvm.dbg.value(metadata, metadata, metadata)

define i32 @func(ptr %0) !dbg !3 {
  call void @llvm.dbg.value(metadata ptr %0, metadata !6, metadata !DIExpression()), !dbg !10
  ret i32 0
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "file.c", directory: "/")
!3 = distinct !DISubprogram(name: "func", scope: !2, file: !2, line: 46, type: !4, scopeLine: 48, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!4 = distinct !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocalVariable(name: "op", arg: 5, scope: !3, file: !2, line: 47, type: !7)
!7 = !DICompositeType(tag: DW_TAG_array_type, size: 2624, elements: !8)
!8 = !{!9}
!9 = !DISubrange(count: 41)
!10 = !DILocation(line: 0, scope: !3)
