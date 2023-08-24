; Test remark output for stack-frame-layout

; ensure basic output works
; RUN: llc -mcpu=corei7 -O1 -pass-remarks-analysis=stack-frame-layout < %s 2>&1 >/dev/null | FileCheck %s

; check additional slots are displayed when stack is not optimized
; RUN: llc -mcpu=corei7 -O0 -pass-remarks-analysis=stack-frame-layout < %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NO_COLORING

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
declare i32 @printf(ptr, ...)

; CHECK: Function: stackSizeWarning
; CHECK: Offset: [SP-88], Type: Variable, Align: 16, Size: 80
; CHECK:    buffer @ frame-diags.c:30
; NO_COLORING: Offset: [SP-168], Type: Variable, Align: 16, Size: 80
; CHECK:    buffer2 @ frame-diags.c:33
define void @stackSizeWarning() {
entry:
  %buffer = alloca [80 x i8], align 16
  %buffer2 = alloca [80 x i8], align 16
  call void @llvm.dbg.declare(metadata ptr %buffer, metadata !25, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %buffer2, metadata !31, metadata !DIExpression()), !dbg !40
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "frame-diags.c", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "dot.c", directory: "")
!4 = !{!5, !6, !10, !13}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Array", file: !3, line: 3, size: 128, elements: !8)
!8 = !{!9, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !7, file: !3, line: 4, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !7, file: !3, line: 5, baseType: !11, size: 32, offset: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Result", file: !3, line: 8, size: 128, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !14, file: !3, line: 9, baseType: !6, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "sum", scope: !14, file: !3, line: 10, baseType: !11, size: 32, offset: 64)
!18 = !{i32 7, !"Dwarf Version", i32 5}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{i32 8, !"PIC Level", i32 2}
!22 = !{i32 7, !"PIE Level", i32 2}
!23 = !{i32 7, !"uwtable", i32 2}
!24 = !{i32 7, !"frame-pointer", i32 2}
!25 = !DILocalVariable(name: "buffer", scope: !26, file: !1, line: 30, type: !32)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 29, column: 3)
!27 = distinct !DISubprogram(name: "stackSizeWarning", scope: !1, file: !1, line: 28, type: !28, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !{!25, !31, !36, !37}
!31 = !DILocalVariable(name: "buffer2", scope: !27, file: !1, line: 33, type: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !33, size: 640, elements: !34)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !{!35}
!35 = !DISubrange(count: 80)
!36 = !DILocalVariable(name: "a", scope: !27, file: !1, line: 34, type: !11)
!37 = !DILocalVariable(name: "b", scope: !27, file: !1, line: 35, type: !38)
!38 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!39 = !DILocation(line: 30, column: 10, scope: !26)
!40 = !DILocation(line: 33, column: 8, scope: !27)
