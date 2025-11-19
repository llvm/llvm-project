; RUN: opt %s -o /dev/null -S 2>&1 | FileCheck %s
;
; The last dbg.declare intrinsic in this file has an illegal DILocation -- this
; needs to pass through the autoupgrade to #dbg_declare process and then get
; caught by the verifier.
;
; CHECK:      invalid #dbg record DILocation
; CHECK-NEXT: #dbg_declare(ptr %1, ![[VAR:[0-9]+]], !DIExpression(), ![[PROG:[0-9]+]])
; CHECK-NEXT: ![[PROG]] = distinct !DISubprogram(name: "IgnoreIntrinsicTest",
; CHECK-NEXT: label %0
; CHECK-NEXT: ptr @IgnoreIntrinsicTest

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define i32 @IgnoreIntrinsicTest() !dbg !10 {
  %1 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %1, metadata !14, metadata !DIExpression()), !dbg !10
  store volatile i32 1, ptr %1, align 4
  %2 = load volatile i32, ptr %1, align 4
  %3 = mul nsw i32 %2, 42
  ret i32 %3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.4 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !5, globals: !5, imports: !5)
!1 = !DIFile(filename: "<unknown>", directory: "/Users/matt/ryan_bug")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !4, file: !1, line: 20, size: 32, align: 32, elements: !6)
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 19, size: 8, align: 8, elements: !5)
!5 = !{}
!6 = !{!7}
!7 = !DIEnumerator(name: "max_frame_size", value: 0)
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "IgnoreIntrinsicTest", linkageName: "IgnoreIntrinsicTest", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !5)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 2, type: !13)
!15 = !DILocation(line: 2, column: 16, scope: !10)
